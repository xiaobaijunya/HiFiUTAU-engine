"""
hifiutau-engine HTTP 服务 — CPU / DML Worker Pool 版本

CPU 模式: 直接在当前进程合成（同原版逻辑）
DML 模式: 启动 N 个独立工作进程，每个预加载 ONNX(DML) 模型，
          通过 Pipe IPC 派发任务，自动选择空闲 worker。

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
import logging
import threading
import multiprocessing
import multiprocessing.connection
from main_onnx import synthesize_audio, preload_all

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
        'dml_workers': 2,
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
            elif key == 'dml_workers':
                config['dml_workers'] = int(val)

    print(f"[配置] 读取 {path}: device={config['device']}, "
          f"dml_workers={config['dml_workers']}")
    return config


# ============================================================================
# DML Worker Pool — 多进程 IPC 任务池
# ============================================================================

class DMLWorkerPool:
    """DML 工作进程池。

    启动 N 个独立进程，每个进程预加载 ONNX(DML) 模型。
    通过 multiprocessing.Pipe 进行双向通信。
    线程安全 — 多个 Flask 线程可同时 acquire 不同 worker。
    """

    def __init__(self, num_workers: int):
        self._num_workers = num_workers
        # 信号量：追踪空闲 worker 数量
        self._available = threading.Semaphore(num_workers)
        self._lock = threading.Lock()
        # worker 状态
        self._idle: set[int] = set()
        self._conns: dict[int, multiprocessing.connection.Connection] = {}
        self._processes: list[multiprocessing.Process] = []

        self._start_workers()

    # ── 进程管理 ────────────────────────────────────────

    def _start_workers(self):
        """启动所有 DML 工作进程，每个进程预加载模型。"""
        print(f"\n{'='*50}")
        print(f"  启动 {self._num_workers} 个 DML 工作进程...")
        print(f"{'='*50}\n")

        for i in range(self._num_workers):
            parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
            p = multiprocessing.Process(
                target=_dml_worker_main,
                args=(child_conn, i),
                daemon=True,
            )
            p.start()
            child_conn.close()  # 父进程不需要 child 端

            self._processes.append(p)
            self._conns[i] = parent_conn
            self._idle.add(i)

            # 等待 worker 发送就绪信号
            ready_msg = parent_conn.recv()
            assert ready_msg == 'READY', f"Worker #{i} 启动失败: {ready_msg}"
            print(f"  [OK] DML Worker #{i+1} 就绪")

        print(f"\n 全部 {self._num_workers} 个 DML Worker 就绪，等待合成请求\n")

    def shutdown(self):
        """安全停止所有 worker。"""
        print("\n[关闭] 正在停止 DML 工作进程...")
        for i, conn in self._conns.items():
            try:
                conn.send('STOP')
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        print("[关闭] 所有 DML 工作进程已停止")

    # ── 任务派发 ────────────────────────────────────────

    def synthesize(self, json_data: dict) -> bytes:
        """获取一个空闲 worker 执行合成，阻塞直到完成。

        Args:
            json_data: OpenUTAU JSON 数据

        Returns:
            WAV bytes

        Raises:
            RuntimeError: 合成失败时抛出
        """
        # 等待空闲 worker（阻塞）
        self._available.acquire()

        idx = -1
        conn = None
        with self._lock:
            idx = next(iter(self._idle))
            self._idle.remove(idx)
            conn = self._conns[idx]

        try:
            # 发送任务
            conn.send(json_data)
            # 接收结果
            result = conn.recv()

            if isinstance(result, Exception):
                raise RuntimeError(
                    f"DML Worker #{idx+1} 合成失败") from result
            return result

        finally:
            with self._lock:
                self._idle.add(idx)
            self._available.release()


def _dml_worker_main(conn: multiprocessing.connection.Connection,
                     worker_id: int):
    """DML 工作进程主函数 — 在子进程中运行。

    预加载模型后进入事件循环，等待 Pipe 上的任务。
    每个进程有独立的全局变量作用域。
    """
    tag = f"[DML Worker #{worker_id+1}]"
    try:
        # ── 预加载 ONNX 模型 (DML) ──
        print(f"{tag} 正在预加载 ONNX 模型 (device=dml)...")
        preload_all(device='dml')
        print(f"{tag} 预加载完成")

        # ── 发送就绪信号 ──
        conn.send('READY')

        # ── 任务循环 ──
        while True:
            msg = conn.recv()
            if msg == 'STOP':
                print(f"{tag} 收到停止信号，退出")
                break

            json_data = msg
            print(f"{tag} 开始合成...")
            t0 = time.time()

            try:
                wav_bytes = synthesize_audio(json_data, device='dml')
                conn.send(wav_bytes)
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

# 全局 pool（由启动代码设置）
_pool: DMLWorkerPool | None = None
_device: str = 'cpu'


@app.route('/synthesize', methods=['POST'])
def receive_json():
    try:
        data = request.get_json()

        with open('test.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))

        if _device == 'dml':
            # DML 模式：通过 Worker Pool 派发
            print('开始合成 (DML Worker Pool)...')
            wav_bytes = _pool.synthesize(data)
        else:
            # CPU 模式：直接当前进程合成
            print('开始合成 (CPU)...')
            wav_bytes = synthesize_audio(data, device='cpu')

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

        num_workers = config['dml_workers']
        _pool = DMLWorkerPool(num_workers)

        try:
            print(f"Server starting on http://localhost:{SERVER_PORT} (DML, "
                  f"{num_workers} workers)")
            print("API endpoints:")
            print("  POST /synthesize - 接收JSON数据并返回二进制wav音频")
            app.run(host='localhost', port=SERVER_PORT, debug=False,
                    threaded=True)
        finally:
            _pool.shutdown()

    else:
        # ── CPU 模式（默认） ──
        import onnxruntime
        print(f"ONNX Runtime 可用设备: "
              f"{onnxruntime.get_available_providers()}")
        print(f"正在预加载 ONNX 模型 (device=cpu)...")
        try:
            preload_all(device='cpu')
        except Exception as e:
            print(f"[WARN] 模型预加载失败（不影响运行，但首次合成会慢）: {e}")
        print("预加载完成")

        print(f"Server starting on http://localhost:{SERVER_PORT} (CPU)")
        print("API endpoints:")
        print("  POST /synthesize - 接收JSON数据并返回二进制wav音频")
        app.run(host='localhost', port=SERVER_PORT, debug=False,
                threaded=True)
