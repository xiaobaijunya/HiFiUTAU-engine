from flask import Flask, request, Response
import json
from main_onnx import synthesize_audio, preload_all

app = Flask(__name__)


@app.route('/synthesize', methods=['POST'])
def receive_json():
    try:
        print('开始合成')
        data = request.get_json()

        # 保存JSON到文件（可选）
        with open('test.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))

        # 调用合成函数，返回二进制wav数据
        wav_bytes = synthesize_audio(data)

        print('合成完成')

        # 直接返回二进制wav数据
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
    # ─── 服务器启动时预加载 ONNX 模型 ───
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else 'dml'
    print(f"正在预加载 ONNX 模型 (device={device})...")
    try:
        preload_all(device=device)
        from tools.hnsep_onnx import preload_hnsep_model
        preload_hnsep_model()
    except Exception as e:
        print(f"[WARN] 模型预加载失败（不影响运行，但首次合成会慢）: {e}")
    print("预加载完成")

    print("Server starting on http://localhost:8000")
    print("API endpoints:")
    print("  POST /synthesize - 接收JSON数据并返回二进制wav音频（隐空间混合拼接）")
    app.run(host='localhost', port=8000, debug=False)