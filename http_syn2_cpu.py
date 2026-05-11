"""
hifisampler HTTP 服务 — CPU 版本
用法:  python http_syn2_cpu.py
      hifiserver_cpu.exe
"""
from flask import Flask, request, Response
import json
import warnings
from main_onnx import synthesize_audio, preload_all

# 屏蔽 Werkzeug 开发服务器警告
warnings.filterwarnings('ignore', message='.*development server.*')

DEVICE = 'cpu'
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
    print(f"正在预加载 ONNX 模型 (device={DEVICE})...")
    try:
        preload_all(device=DEVICE)
        from tools.hnsep_onnx import preload_hnsep_model
        preload_hnsep_model()
    except Exception as e:
        print(f"[WARN] 模型预加载失败（不影响运行，但首次合成会慢）: {e}")
    print("预加载完成")

    print(f"Server starting on http://localhost:8000 (CPU)")
    print("API endpoints:")
    print("  POST /synthesize - 接收JSON数据并返回二进制wav音频（隐空间混合拼接）")
    app.run(host='localhost', port=8000, debug=False)
