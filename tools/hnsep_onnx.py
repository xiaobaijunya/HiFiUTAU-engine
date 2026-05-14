"""
HN-SEP ONNX 推理模块

继承 BaseHnsep，差异只在 ONNX 模型加载和推理。
公共逻辑（breath/tension 后处理）在 base_hnsep.py 中。
"""

import os as _os
import numpy as np
import onnxruntime
from tools.base_hnsep import BaseHnsep


class OnnxHnsep(BaseHnsep):
    """HN-SEP 谐波/噪声分离器 — ONNX Runtime 版。"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Args:
            model_path: .onnx 模型路径，默认 hnsep_onnx/hnsep_VR_44.1k_hop512_2024.05.onnx
            device:     'cpu' 或 'cuda'（跳过 DML，LSTM 有 bug）
        """
        if model_path is None:
            model_path = _os.path.join(
                "hnsep_onnx", "hnsep_VR_44.1k_hop512_2024.05.onnx")

        available = onnxruntime.get_available_providers()
        dl = device.lower()
        if dl in ('cuda', 'gpu', 'tensorrt', 'trt'):
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if 'CUDAExecutionProvider' in available
                         else ['CPUExecutionProvider'])
        else:
            providers = ['CPUExecutionProvider']

        print(f"加载 HN-SEP ONNX 模型: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        print(f'HN-SEP ONNX 模型已加载, providers: {self.session.get_providers()}')

    def separate(self, waveform: np.ndarray) -> tuple:
        """分离音频为谐波和噪声分量。"""
        waveform = self._ensure_1d(waveform)
        wav_input = waveform.reshape(1, -1)
        harmonic, noise = self.session.run(
            ['harmonic', 'noise'], {'waveform': wav_input})
        return harmonic[0], noise[0]


# ═══════════════════════════════════════════════════════════════
#  全局单例 + 向后兼容函数
# ═══════════════════════════════════════════════════════════════

_global_hnsep_instance: OnnxHnsep | None = None


def get_global_hnsep_session(model_path: str = None, device: str = 'cpu'):
    """获取全局 ONNX HN-SEP 实例（单例）。"""
    global _global_hnsep_instance
    if _global_hnsep_instance is None:
        _global_hnsep_instance = OnnxHnsep(model_path, device)
    return _global_hnsep_instance.session


def preload_hnsep_model(model_path: str = None, device: str = 'cpu'):
    """预加载 HN-SEP ONNX 模型。"""
    try:
        global _global_hnsep_instance
        _global_hnsep_instance = OnnxHnsep(model_path, device)
        print("[OK] HN-SEP 模型预加载成功")
        return True
    except Exception as e:
        print(f"[WARN] HN-SEP 模型预加载失败: {e}")
        return False


def hnsep_separate(waveform: np.ndarray, session=None) -> tuple:
    """向后兼容：使用指定会话或全局实例分离谐波/噪声。

    支持直接传入 ONNX InferenceSession（post_process.py 的用法）。
    """
    if session is not None and not isinstance(session, OnnxHnsep):
        # 直接使用传入的 ONNX InferenceSession
        wav = BaseHnsep._ensure_1d(waveform)
        wav_input = wav.reshape(1, -1)
        harmonic, noise = session.run(['harmonic', 'noise'], {'waveform': wav_input})
        return harmonic[0], noise[0]
    global _global_hnsep_instance
    if _global_hnsep_instance is None:
        _global_hnsep_instance = OnnxHnsep()
    return _global_hnsep_instance.separate(waveform)


def apply_breath_tension(waveform, breath=100, voicing=100, tension=0, session=None):
    """向后兼容：应用 breath/tension 后处理。"""
    global _global_hnsep_instance
    if _global_hnsep_instance is None:
        _global_hnsep_instance = OnnxHnsep()
    return _global_hnsep_instance.apply_breath_tension(
        waveform, breath=breath, voicing=voicing, tension=tension)
    processed = apply_breath_tension(test_wav, breath=150, tension=50)
    print(f"处理后音频: {processed.shape}")

    print("[OK] HN-SEP ONNX 模块测试通过")
