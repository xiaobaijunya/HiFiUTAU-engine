from tools.hifigan import set_global_ort_session
import onnxruntime
from pathlib import Path

# 全局变量存储预加载的模型
ort_session = None
model_path = None

def preload_hifigan_model(model_file_path):
    """预加载HiFiGAN模型"""
    global ort_session, model_path

    model_path = Path(model_file_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if model_path.suffix == '.onnx':
        print(f"预加载ONNX模型: {model_path}")

        # 选择ONNX运行时provider
        available_providers = onnxruntime.get_available_providers()
        print(f'Available providers: {available_providers}')
        preferred_providers = []
        if 'DmlExecutionProvider' in available_providers:
            preferred_providers.append('DmlExecutionProvider')
        elif 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        preferred_providers.append('CPUExecutionProvider')

        ort_session = onnxruntime.InferenceSession(
            str(model_path), providers=preferred_providers)
        print(f"✅ ONNX模型加载成功，使用providers: {ort_session.get_providers()}")

        # 将预加载的会话设置为全局会话，供其他模块使用
        set_global_ort_session(ort_session)
        print(f"✅ 已设置全局ONNX会话")

    else:
        print(f"⚠️  模型格式 {model_path.suffix} 不支持预加载，将在运行时加载")