"""
hifiutau-engine HTTP 服务 — PyTorch 推理版

使用原生 PyTorch 进行合成，替代复杂的 ONNX Runtime + TensorRT/DirectML 链路。
用法:  python http_syn2_pytorch.py
       hifiserver_pytorch.exe
"""
from flask import Flask, request, Response
import json
import base64
import logging
import os
import waitress
from concurrent.futures import ThreadPoolExecutor
from main_pytorch import (
    synthesize_audio, synthesize_mel, synthesize_hnsep, synthesize_post,
    preload_all,
)

DEVICE = 'cuda'
CONFIG_FILE = 'run_config.txt'
app = Flask(__name__)

# 全局池（由启动代码设置）
_hifigan_pool: ThreadPoolExecutor | None = None  # /syn_mel + 旧 /synthesize
_hnsep_pool: ThreadPoolExecutor | None = None    # /syn_hnsep


def read_worker_config(path: str = CONFIG_FILE) -> dict:
    """读取并发配置（hifigan_workers / hnsep_workers / infer_threads）。"""
    config = {'hifigan_workers': 2, 'hnsep_workers': 1, 'infer_threads': 1}
    if os.path.exists(path):
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
                if key == 'hifigan_workers':
                    config['hifigan_workers'] = int(val)
                elif key == 'hnsep_workers':
                    config['hnsep_workers'] = int(val)
                elif key == 'infer_threads':
                    config['infer_threads'] = int(val)
    print(f"[配置] hifigan_workers={config['hifigan_workers']}, "
          f"hnsep_workers={config['hnsep_workers']}, "
          f"infer_threads={config['infer_threads']}")
    return config


@app.route('/synthesize', methods=['POST'])
def receive_json():
    """旧完整合成（兼容旧 CustomRenderer）。走统一的 hifigan 线程池。"""
    try:
        print('开始合成 (full)...')
        data = request.get_json()

        with open('test.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))

        wav_bytes = _hifigan_pool.submit(
            _run_segment, '', data, DEVICE).result()

        print('合成完成')

        return Response(
            wav_bytes,
            mimetype='audio/wav',
            headers={
                'Content-Length': str(len(wav_bytes))
            }
        )

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

    mel → hifigan 线程池；hnsep → hnsep 线程池；post → 直接计算（轻量）。
    """
    try:
        data = request.get_json()
        if mode == 'mel':
            result = _hifigan_pool.submit(
                _run_segment, 'mel', data, DEVICE).result()
        elif mode == 'hnsep':
            result = _hnsep_pool.submit(
                _run_segment, 'hnsep', data, DEVICE).result()
        else:  # post → 轻量，直接计算（不进池）
            result = _run_segment('post', data, DEVICE)

        if mode == 'hnsep':
            return _format_hnsep_response(result)
        return _format_single_response(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return Response(str(e), status=400, mimetype='text/plain')


if __name__ == '__main__':
    import os as _os

    # 从环境变量读取优化选项
    _compile = _os.environ.get('HIFIUTAU_ENGINE_COMPILE', '0') == '1'
    _fp16 = _os.environ.get('HIFIUTAU_ENGINE_FP16', '0') == '1'
    _cfg = read_worker_config()

    # 限制 torch 推理线程数（避免挤占资源）
    try:
        import torch
        torch.set_num_threads(max(1, int(_cfg['infer_threads'])))
        print(f"[配置] torch.set_num_threads({_cfg['infer_threads']})")
    except Exception:
        pass

    _hifigan_pool = ThreadPoolExecutor(
        max_workers=_cfg['hifigan_workers'], thread_name_prefix='ht-hifigan')
    _hnsep_pool = ThreadPoolExecutor(
        max_workers=_cfg['hnsep_workers'], thread_name_prefix='ht-hnsep')

    print(f"正在预加载 PyTorch 模型 (device={DEVICE})...")
    print(f"  优化: compile={_compile}, fp16={_fp16}")
    print(f"  提示: 设置环境变量 HIFIUTAU_ENGINE_COMPILE=1 启用 torch.compile")
    print(f"        设置环境变量 HIFIUTAU_ENGINE_FP16=1 启用 FP16 推理")

    try:
        preload_all(device=DEVICE, compile_model=_compile, fp16=_fp16)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("""
提示：需要将 PyTorch checkpoint 放到正确位置。
方法1: 将 pc_nsf_hifigan_44.1k_hop512_128bin_2025.02 文件夹（含 model.ckpt 和 config.json）
       复制到工作目录
方法2: 设置环境变量 HIFIUTAU_ENGINE_CKPT 指向 model.ckpt 路径
       $env:HIFIUTAU_ENGINE_CKPT = "D:\\models\\pc_nsf_hifigan\\model.ckpt"
""")
        exit(1)
    except Exception as e:
        print(f"[WARN] 模型预加载失败（不影响运行，但首次合成会慢）: {e}")
    print("预加载完成")

    print(f"Server starting on http://localhost:8000 (PyTorch, device={DEVICE}, waitress)")
    print(f"  Pool: hifigan={_cfg['hifigan_workers']}, hnsep={_cfg['hnsep_workers']}, infer_threads={_cfg['infer_threads']}")
    print("API endpoints:")
    print("  POST /synthesize  - 旧完整合成 (hifigan池)")
    print("  POST /syn_mel     - 分段1: mel拼接+变调+HiFi-GAN (hifigan池)")
    print("  POST /syn_hnsep   - 分段2: HN-SEP 气声/谐波分离 (hnsep池)")
    print("  POST /syn_post    - 分段3: 参数应用 (进程内CPU直算)")
    waitress.serve(app, host='localhost', port=8000, threads=8)
