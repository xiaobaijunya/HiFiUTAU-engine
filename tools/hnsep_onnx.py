"""
HN-SEP ONNX 推理模块

使用 ONNX 版本的 HN-SEP 模型对音频进行谐波/噪声分离。
模型输入: waveform (batch_size, n_samples) - 2D float32
模型输出: harmonic (batch_size, n_samples), noise (batch_size, n_samples)
"""

import numpy as np
import onnxruntime


# 全局 ONNX 会话缓存
_global_hnsep_session = None


def get_global_hnsep_session(model_path: str = None):
    """获取全局 HN-SEP ONNX 会话（单例）。

    注意: HN-SEP 模型包含 LSTM op，DirectML 有 bug（运行时崩溃），
          因此强制使用 CPU。（实测 DML 首次推理 9s 且第二次必崩）
    """
    global _global_hnsep_session
    if _global_hnsep_session is None:
        if model_path is None:
            model_path = r"hnsep_onnx\hnsep_VR_44.1k_hop512_2024.05.onnx"
        print(f"加载 HN-SEP ONNX 模型: {model_path}")
        _global_hnsep_session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        print(f'HN-SEP ONNX 模型已加载, providers: {_global_hnsep_session.get_providers()}')
    return _global_hnsep_session


def preload_hnsep_model(model_path: str = None):
    """预加载 HN-SEP ONNX 模型。"""
    try:
        get_global_hnsep_session(model_path)
        print("[OK] HN-SEP 模型预加载成功")
        return True
    except Exception as e:
        print(f"[WARN] HN-SEP 模型预加载失败: {e}")
        return False


def hnsep_separate(waveform: np.ndarray, session=None) -> tuple:
    """
    使用 HN-SEP ONNX 模型分离音频为谐波和噪声分量。

    Args:
        waveform: np.ndarray, shape (samples,) 或 (1, samples), float32
        session: ONNX InferenceSession, 如果为 None 则使用全局会话

    Returns:
        harmonic: np.ndarray, shape (samples,) - 谐波分量
        noise: np.ndarray, shape (samples,) - 噪声/气息分量
    """
    if session is None:
        session = get_global_hnsep_session()

    # 确保输入为 (1, n_samples) 2D 格式
    if waveform.ndim == 1:
        waveform_input = waveform.reshape(1, -1).astype(np.float32)
    elif waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform_input = waveform.astype(np.float32)
    else:
        waveform_input = waveform.reshape(1, -1).astype(np.float32)

    harmonic, noise = session.run(['harmonic', 'noise'], {'waveform': waveform_input})

    # 返回 1D 数组
    return harmonic[0], noise[0]


def apply_breath_tension(
    waveform: np.ndarray,
    breath: float = 100.0,
    voicing: float = 100.0,
    tension: float = 0.0,
    session=None
) -> np.ndarray:
    """
    使用 HN-SEP 对音频应用 breath（气声）、voicing（发声）、tension（张力）参数。

    原理（参考 hifiserver.py）:
      1. 使用 HN-SEP 分离谐波和噪声
      2. breath 控制噪声（气息）电平: noise_out = noise * (breath / 100)
      3. voicing 控制谐波（发声）电平: harmonic_out = harmonic * (voicing / 100)
      4. tension 控制谐波部分的频谱倾斜（预加重滤波）
      5. 最终音频 = 处理后的噪声 + 处理后的谐波

    Args:
        waveform: np.ndarray, shape (samples,) - 输入音频
        breath: float, 气息量 (0-500, 默认 100 = 原始)
        voicing: float, 发声量 (0-150, 默认 100 = 原始)
        tension: float, 张力 (-100 ~ 100, 默认 0 = 原始)
        session: ONNX InferenceSession

    Returns:
        np.ndarray, shape (samples,) - 处理后的音频
    """
    # 如果参数都是默认值，直接返回
    if (abs(breath - 100) < 0.5 and abs(voicing - 100) < 0.5
            and abs(tension) < 0.5):
        return waveform

    harmonic, noise = hnsep_separate(waveform, session)

    # 处理 breath（气息）- 缩放噪声分量
    breath = np.clip(breath, 0, 500)
    noise_out = noise * (breath / 100.0)

    # 处理 voicing（发声）- 缩放谐波分量
    voicing = np.clip(voicing, 0, 150)
    harmonic_scaled = harmonic * (voicing / 100.0)

    # 处理 tension（张力）- 对谐波部分做预加重/去加重
    tension = np.clip(tension, -100, 100)
    if abs(tension) > 0.5:
        harmonic_out = _apply_tension_filter(harmonic_scaled, tension)
    else:
        harmonic_out = harmonic_scaled

    # 混合
    result = noise_out + harmonic_out

    # 保持原始响度水平
    original_rms = np.sqrt(np.mean(waveform ** 2))
    result_rms = np.sqrt(np.mean(result ** 2))
    if result_rms > 1e-8:
        result = result * (original_rms / result_rms)

    return result


def _apply_tension_filter(waveform: np.ndarray, tension: float) -> np.ndarray:
    """
    对音频应用张力（频谱倾斜）滤波器。

    参考 pre_emphasis_base_tension:
      - tension > 0: 增强高频（预加重）
      - tension < 0: 减弱高频（去加重）
    
    使用简单的一阶 FIR 滤波器实现:
      y[n] = x[n] - alpha * x[n-1]
    其中 alpha 控制倾斜程度。
    """
    # tension: -100 ~ 100, 映射到 alpha: -0.97 ~ 0.97
    alpha = tension / 100.0 * 0.97

    if abs(alpha) < 0.01:
        return waveform

    # 应用 FIR 滤波器
    filtered = np.zeros_like(waveform)
    filtered[0] = waveform[0]
    for i in range(1, len(waveform)):
        filtered[i] = waveform[i] - alpha * waveform[i - 1]

    # 保持能量水平
    original_rms = np.sqrt(np.mean(waveform ** 2))
    filtered_rms = np.sqrt(np.mean(filtered ** 2))
    if filtered_rms > 1e-8:
        filtered = filtered * (original_rms / filtered_rms)

    return filtered


if __name__ == '__main__':
    # 简单测试
    import soundfile as sf
    import os

    # 预加载模型
    preload_hnsep_model()

    # 生成测试音频（正弦波 + 噪声）
    sr = 44100
    t = np.linspace(0, 1, sr)
    test_wav = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(sr)
    test_wav = test_wav.astype(np.float32)

    harmonic, noise = hnsep_separate(test_wav)
    print(f"分离成功! 谐波: {harmonic.shape}, 噪声: {noise.shape}")

    # 测试 breath/tension 处理
    processed = apply_breath_tension(test_wav, breath=150, tension=50)
    print(f"处理后音频: {processed.shape}")

    print("[OK] HN-SEP ONNX 模块测试通过")
