"""
hifiutau-engine HTTP 服务 — PyTorch 推理版

使用原生 PyTorch 进行合成，替代复杂的 ONNX Runtime + TensorRT/DirectML 链路。
用法:  python http_syn2_pytorch.py
       hifiserver_pytorch.exe
"""
from flask import Flask, request, Response
import json
import logging
import waitress
from main_pytorch import synthesize_audio, preload_all

DEVICE = 'cuda'
app = Flask(__name__)


@app.route('/synthesize', methods=['POST'])
def receive_json():
    try:
        print('开始合成')
        data = request.get_json()

        with open('test.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))

        wav_bytes = synthesize_audio(data, device=DEVICE)

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


if __name__ == '__main__':
    import os as _os

    # torch.compile 默认开启（可设 HIFIUTAU_ENGINE_COMPILE=0 关闭）
    _compile = _os.environ.get('HIFIUTAU_ENGINE_COMPILE', '1') == '1'

    print(f"正在预加载 PyTorch 模型 (device={DEVICE}, compile={_compile})...")

    try:
        preload_all(device=DEVICE, compile_model=_compile)
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
    print("API endpoints:")
    print("  POST /synthesize - 接收JSON数据并返回二进制wav音频（隐空间混合拼接）")
    waitress.serve(app, host='localhost', port=8000, threads=8)
