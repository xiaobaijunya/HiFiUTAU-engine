"""
hifiutau-engine HTTP 服务 — 统一阶段池版本（CPU=线程池 / DML=进程池）

CPU 模式: 模型在主进程预加载，通过线程池限制并发模型推理数量（避免挤占资源）。
DML 模式: 独立工作进程，按池类型预加载对应模型。
          - hifigan 池: 仅 splicer（/syn_mel）
          - hnsep   池: 仅 hnsep（/syn_hnsep）
          - /syn_post 直接在进程内用 CPU 计算（轻量，不占 GPU worker）

并发配置（run_config.txt）:
  device          cpu 或 dml
  hifigan_workers hifigan/完整合成 并发数
  hnsep_workers   hnsep 并发数
  infer_threads   单个模型推理线程数（ONNX intra_op）

用法:
  python http_syn2_cpu.py              # 从 run_config.txt 读取配置
  hifiserver_cpu.exe                   # PyInstaller 打包

端口: 始终为 8000（单入口）
"""
from flask import Flask, request, Response
import json
import os
import sys
import time
import base64
import logging
import threading
import multiprocessing
import multiprocessing.connection
from concurrent.futures import ThreadPoolExecutor
import waitress
from main_onnx import (synthesize_audio, synthesize_mel, synthesize_hnsep,
                       synthesize_post, preload_all)

# 屏蔽 Werkzeug 开发服务器警告
logging.getLogger('werkzeug').setLevel(logging.ERROR)

CONFIG_FILE = 'run_config.txt'
SERVER_PORT = 8000


# ============================================================================
# 配置读取
# ============================================================================

def read_config(path: str = CONFIG_FILE) -> dict:
    """读取 run_config.txt，返回配置字典。"""
    config = {
        'device': 'cpu',
        'hifigan_workers': 2,      # hifigan/完整合成 并发数（CPU=线程池，DML=进程池）
        'hnsep_workers': 1,        # hnsep 并发数
        'infer_threads': 1,        # 单个模型推理线程数（ONNX intra_op）
    }
    if not os.path.exists(path):
        print(f"[配置] {path} 不存在，使用默认配置: {config}")
        return config

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip().lower()
            val = val.strip()
            if key == 'device':
                config['device'] = val.lower()
            elif key == 'hifigan_workers':
                config['hifigan_workers'] = int(val)
            elif key == 'hnsep_workers':
                config['hnsep_workers'] = int(val)
            elif key == 'infer_threads':
                config['infer_threads'] = int(val)

    print(f"[配置] 读取 {path}: device={config['device']}, "
          f"hifigan_workers={config['hifigan_workers']}, "
          f"hnsep_workers={config['hnsep_workers']}, "
          f"infer_threads={config['infer_threads']}")
    return config


# ============================================================================
# 统一阶段池 — CPU 用线程池，DML 用进程池
# ============================================================================

class StagePool:
    """统一阶段池。

    CPU 模式: 线程池，限制并发模型推理数量（避免挤占资源）。
    DML 模式: 进程池，每个进程按 pool_type 预加载对应模型。

    pool_type:
      'hifigan' → 处理 /syn_mel（仅 splicer）
      'hnsep'   → 处理 /syn_hnsep（仅 hnsep）
    """

    def __init__(self, num_workers: int, pool_type: str = 'hifigan',
                 device: str = 'cpu', infer_threads: int = 1):
        self._num_workers = max(1, int(num_workers))
        self._pool_type = pool_type
        self._device = device
        self._infer_threads = max(1, int(infer_threads))
        self._lock = threading.Lock()
        self._executor = None
        self._conns: dict[int, multiprocessing.connection.Connection] = {}
        self._processes: list[multiprocessing.Process] = []

        if device == 'dml':
            self._available = threading.Semaphore(self._num_workers)
            self._idle: set[int] = set()
            self._start_processes()
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers,
                thread_name_prefix=f'ht-{pool_type}')
            print(f"[池] {pool_type} 线程池 {self._num_workers} 个 worker 就绪")

    # ── DML 进程管理 ────────────────────────────────────

    def _start_processes(self):
        """启动 DML 工作进程，按 pool_type 预加载对应模型。"""
        tag = {'hifigan': 'hifigan', 'hnsep': 'hnsep'}.get(
            self._pool_type, self._pool_type)
        print(f"\n{'='*50}")
        print(f"  启动 {self._num_workers} 个 {tag} DML 工作进程...")
        print(f"{'='*50}\n")

        for i in range(self._num_workers):
            parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
            p = multiprocessing.Process(
                target=_dml_worker_main,
                args=(child_conn, i, self._pool_type, self._device,
                      self._infer_threads),
                daemon=True,
            )
            p.start()
            child_conn.close()

            self._processes.append(p)
            self._conns[i] = parent_conn
            self._idle.add(i)

            ready_msg = parent_conn.recv()
            assert ready_msg == 'READY', f"{tag} Worker #{i+1} 启动失败: {ready_msg}"
            print(f"  [OK] {tag} Worker #{i+1} 就绪")

        print(f"\n 全部 {self._num_workers} 个 {tag} Worker 就绪，等待合成请求\n")

    # ── 任务派发 ────────────────────────────────────────

    def synthesize(self, mode: str, payload):
        """派发任务并阻塞等待结果。

        Args:
            mode:    'mel' | 'hnsep' | 'post'
            payload: 请求数据（dict）

        Returns:
            合成结果：WAV bytes 或 dict（hnsep 分离结果）
        """
        if self._executor is not None:
            # CPU 线程池：_run_segment 在池线程内执行（模型已在主进程预加载）
            return self._executor.submit(
                _run_segment, mode, payload, 'cpu').result()

        # DML 进程池
        self._available.acquire()
        idx = -1
        conn = None
        with self._lock:
            idx = next(iter(self._idle))
            self._idle.remove(idx)
            conn = self._conns[idx]
        try:
            conn.send((mode, payload))
            result = conn.recv()
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"DML Worker #{idx+1} 合成失败") from result
            return result
        finally:
            with self._lock:
                self._idle.add(idx)
            self._available.release()

    def shutdown(self):
        """安全停止池。"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            print(f"[池] {self._pool_type} 线程池已关闭")
            return
        print(f"\n[关闭] 正在停止 {self._pool_type} DML 工作进程...")
        for conn in self._conns.values():
            try:
                conn.send('STOP')
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        print(f"[关闭] {self._pool_type} 进程池已停止")


def _preload_by_type(pool_type: str, device: str, infer_threads: int = 1):
    """按池类型预加载模型：
      'full'   → splicer + hnsep（旧完整合成 /synthesize）
      'hifigan'→ 仅 splicer（/syn_mel）
      'hnsep'  → 仅 hnsep（/syn_hnsep）
    """
    from main_onnx import preload_splicer, preload_hnsep, preload_all
    if pool_type == 'hnsep':
        preload_hnsep(device, infer_threads=infer_threads)
    elif pool_type == 'hifigan':
        preload_splicer(device, infer_threads=infer_threads)
    else:
        preload_all(device, infer_threads=infer_threads)


def _dml_worker_main(conn: multiprocessing.connection.Connection,
                     worker_id: int,
                     pool_type: str = 'hifigan',
                     device: str = 'dml',
                     infer_threads: int = 1):
    """DML 工作进程主函数 — 在子进程中运行。

    按 pool_type 预加载对应模型后进入事件循环，等待 Pipe 上的任务。
    每个进程有独立的全局变量作用域。
    """
    tag = f"[{pool_type} Worker #{worker_id+1}]"
    try:
        # ── 按池类型预加载 ONNX 模型 (DML) ──
        print(f"{tag} 正在预加载 ONNX 模型 (device={device})...")
        _preload_by_type(pool_type, device, infer_threads)
        print(f"{tag} 预加载完成")

        # ── 发送就绪信号 ──
        conn.send('READY')

        # ── 任务循环 ──
        while True:
            msg = conn.recv()
            if msg == 'STOP':
                print(f"{tag} 收到停止信号，退出")
                break

            mode, json_data = msg
            print(f"{tag} 开始合成 (mode={mode or 'full'})...")
            t0 = time.time()

            try:
                # hifigan 池处理旧完整合成时懒加载 hnsep（不挤占 /syn_mel 内存）
                if mode == '' and pool_type == 'hifigan':
                    from main_onnx import get_hnsep_session, preload_hnsep
                    if get_hnsep_session() is None:
                        print(f"{tag} 懒加载 HN-SEP...")
                        preload_hnsep(device, infer_threads=infer_threads)
                result = _run_segment(mode, json_data, device)
                conn.send(result)
                print(f"{tag} 合成完成 ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"{tag} 合成出错: {e}")
                import traceback
                traceback.print_exc()
                conn.send(e)

    except Exception as e:
        print(f"{tag} 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        # 告诉父进程启动失败
        try:
            conn.send(f'ERROR: {e}')
        except Exception:
            pass
        raise


# ============================================================================
# Flask 应用
# ============================================================================

app = Flask(__name__)

# 全局池（由启动代码设置）
_hifigan_pool: StagePool | None = None  # /syn_mel + 旧 /synthesize
_hnsep_pool: StagePool | None = None    # /syn_hnsep
_device: str = 'cpu'


@app.route('/synthesize', methods=['POST'])
def receive_json():
    """旧完整合成（兼容旧 CustomRenderer）。走统一的 hifigan 池（CPU=线程池，DML=进程池）。"""
    try:
        data = request.get_json()

        # with open('test.json', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(data, indent=2, ensure_ascii=False))

        print('开始合成 (full)...')
        wav_bytes = _hifigan_pool.synthesize('', data)

        print('合成完成')

        return Response(
            wav_bytes,
            mimetype='audio/wav',
            headers={'Content-Length': str(len(wav_bytes))}
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return Response(str(e), status=400, mimetype='text/plain')


# ============================================================================
# 分段合成端点（HiFiUTAU Local 渲染器）
# ============================================================================

def _extract_wav(data: dict, key: str = 'wav', optional: bool = False):
    """从请求中提取 wav 字节（字段为绝对路径，本地直读）。"""
    v = data.get(key)
    if v is None:
        if optional:
            return None
        raise ValueError(f"缺少字段 '{key}'")
    with open(v, 'rb') as f:
        return f.read()


def _run_segment(mode: str, data: dict, device: str):
    """执行一个合成段，返回 bytes（wav）或 dict（hnsep 结果）。

    引擎不落盘，合成结果统一回传（written=False）。
    供 CPU 直接调用与 DML worker 共用。
    """
    if mode == 'mel':
        return synthesize_mel(data, device=device)

    if mode == 'hnsep':
        wav_bytes = _extract_wav(data, 'wav')
        return synthesize_hnsep(wav_bytes, device=device)

    if mode == 'post':
        wav_bytes = _extract_wav(data, 'wav', optional=True)
        harmonic_bytes = _extract_wav(data, 'harmonic', optional=True)
        noise_bytes = _extract_wav(data, 'noise', optional=True)
        return synthesize_post(
            data, wav_bytes=wav_bytes,
            harmonic_bytes=harmonic_bytes, noise_bytes=noise_bytes,
            device=device)

    # 空 mode → 旧完整合成（兼容旧 CustomRenderer）
    return synthesize_audio(data, device=device)


def _format_single_response(result):
    """mel/post 结果 (bytes, written) → Flask Response。"""
    data, written = result
    if written:
        return Response(
            json.dumps({'ok': True, 'written': True}),
            mimetype='application/json')
    # 本地合成：直接返回原始 wav 字节，不压缩
    return Response(
        data, mimetype='audio/wav',
        headers={
            'Content-Length': str(len(data)),
            'X-HiFiUTAU-Written': 'false',
        })


def _format_hnsep_response(result):
    """hnsep 结果 (harmonic, noise, written_h, written_n) → Flask Response。"""
    hb, nb, w_h, w_n = result
    resp = {'ok': True, 'written_harmonic': w_h, 'written_noise': w_n}
    if not w_h:
        resp['harmonic_b64'] = base64.b64encode(hb).decode()
    if not w_n:
        resp['noise_b64'] = base64.b64encode(nb).decode()
    return Response(json.dumps(resp), mimetype='application/json')


def _segment_endpoint(mode: str):
    """三个分段端点的公共处理。

    DML 模式:
      mel  → hifigan 池（仅 splicer）
      hnsep→ hnsep 池（仅 hnsep）
      post → 直接在 Flask 进程内用 CPU 计算（轻量，不占 GPU worker）
    CPU 模式: 全部直接计算（无池）
    """
    try:
        data = request.get_json()
        if mode == 'mel':
            result = _hifigan_pool.synthesize('mel', data)
        elif mode == 'hnsep':
            result = _hnsep_pool.synthesize('hnsep', data)
        else:  # post → 轻量，直接 CPU 计算（不进池）
            result = _run_segment('post', data, 'cpu')

        if mode == 'hnsep':
            return _format_hnsep_response(result)
        return _format_single_response(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return Response(str(e), status=400, mimetype='text/plain')


@app.route('/syn_mel', methods=['POST'])
def syn_mel():
    return _segment_endpoint('mel')


@app.route('/syn_hnsep', methods=['POST'])
def syn_hnsep():
    return _segment_endpoint('hnsep')


@app.route('/syn_post', methods=['POST'])
def syn_post():
    return _segment_endpoint('post')


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    # PyInstaller 打包后 multiprocessing 需要此调用防止递归启动
    multiprocessing.freeze_support()

    config = read_config()
    _device = config['device']

    if _device == 'dml':
        # ── DML 模式 ──
        # 检查 DmlExecutionProvider 是否可用（不可用直接报错，不自动回退）
        import onnxruntime
        available = onnxruntime.get_available_providers()
        if 'DmlExecutionProvider' not in available:
            print(f"[FATAL] 配置为 device=dml，但 DmlExecutionProvider 不可用！")
            print(f"        当前可用 Providers: {available}")
            print(f"        请安装 DirectML: pip install onnxruntime-directml")
            sys.exit(1)

        _hifigan_pool = StagePool(config['hifigan_workers'], 'hifigan',
                                  'dml', config['infer_threads'])
        _hnsep_pool = StagePool(config['hnsep_workers'], 'hnsep',
                                'dml', config['infer_threads'])

        try:
            print(f"Server starting on http://localhost:{SERVER_PORT} (DML, waitress)")
            print(f"  Pool: hifigan={config['hifigan_workers']}, hnsep={config['hnsep_workers']}, infer_threads={config['infer_threads']}")
            print("API endpoints:")
            print("  POST /synthesize  - 旧完整合成 (hifigan池)")
            print("  POST /syn_mel     - 分段1: mel拼接+变调+HiFi-GAN (hifigan池)")
            print("  POST /syn_hnsep   - 分段2: HN-SEP 气声/谐波分离 (hnsep池)")
            print("  POST /syn_post    - 分段3: 参数应用 (进程内CPU直算)")
            waitress.serve(app, host='localhost', port=SERVER_PORT,
                           threads=8)
        finally:
            _hifigan_pool.shutdown()
            _hnsep_pool.shutdown()

    else:
        # ── CPU 模式（默认） ──
        import onnxruntime
        print(f"ONNX Runtime 可用设备: "
              f"{onnxruntime.get_available_providers()}")
        print(f"正在预加载 ONNX 模型 (device=cpu, infer_threads={config['infer_threads']})...")
        try:
            preload_all(device='cpu', infer_threads=config['infer_threads'])
        except Exception as e:
            print(f"[WARN] 模型预加载失败（不影响运行，但首次合成会慢）: {e}")
        print("预加载完成")

        _hifigan_pool = StagePool(config['hifigan_workers'], 'hifigan',
                                  'cpu', config['infer_threads'])
        _hnsep_pool = StagePool(config['hnsep_workers'], 'hnsep',
                                'cpu', config['infer_threads'])

        print(f"Server starting on http://localhost:{SERVER_PORT} (CPU, waitress)")
        print(f"  Pool: hifigan={config['hifigan_workers']}, hnsep={config['hnsep_workers']}, infer_threads={config['infer_threads']}")
        print("API endpoints:")
        print("  POST /synthesize  - 旧完整合成 (hifigan池)")
        print("  POST /syn_mel     - 分段1: mel拼接+变调+HiFi-GAN (hifigan池)")
        print("  POST /syn_hnsep   - 分段2: HN-SEP 气声/谐波分离 (hnsep池)")
        print("  POST /syn_post    - 分段3: 参数应用 (进程内CPU直算)")
        waitress.serve(app, host='localhost', port=SERVER_PORT,
                       threads=8)
