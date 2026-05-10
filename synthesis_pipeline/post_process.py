"""
HN-SEP 后处理 — 动态 breath / tension / voicing 参数应用。
"""
import numpy as np

from synthesis_pipeline.utils import interp_to_len
from synthesis_pipeline.tension_filter import apply_dynamic_tension


def apply_hnsep_postprocess(
    wav: np.ndarray,
    breath_array: np.ndarray,
    tension_array: np.ndarray,
    voicing_array: np.ndarray,
    sr: int,
    hnsep_session,
) -> np.ndarray:
    """对合成后的音频应用 HN-SEP breath/tension/voicing 后处理。

    Args:
        wav:             合成音频 (samples,)
        breath_array:    Dynamic_hop 帧间距的 breath 值 (-100~100)
        tension_array:   Dynamic_hop 帧间距的 tension 值 (-100~100)
        voicing_array:   Dynamic_hop 帧间距的 voicing 值 (0~500)
        sr:              采样率
        hnsep_session:   HN-SEP ONNX 推理会话

    Returns:
        处理后的音频
    """
    # 检查是否需要处理
    has_breath = len(breath_array) > 0
    has_tension = len(tension_array) > 0
    has_voicing = len(voicing_array) > 0

    if not (has_breath or has_tension or has_voicing):
        return wav

    need_breath = has_breath and not np.allclose(breath_array, 0, atol=0.5)
    need_tension = has_tension and not np.allclose(tension_array, 0, atol=0.5)
    need_voicing = has_voicing and not np.allclose(voicing_array, 100, rtol=0.05)

    if not (need_breath or need_tension or need_voicing):
        return wav

    print("HN-SEP 后处理: 分离谐波/噪声 + 动态 breath/tension/voicing...")
    from tools.hnsep_onnx import hnsep_separate

    harmonic, noise = hnsep_separate(wav, hnsep_session)
    n_samples = len(wav)

    # breath — 感知压缩映射
    if need_breath:
        breath_raw = interp_to_len(breath_array, n_samples)
        pos = breath_raw >= 0
        neg = ~pos
        noise_gain = np.empty(n_samples, dtype=np.float32)
        noise_gain[pos] = 1.0 + (breath_raw[pos] / 100.0) ** 0.7 * 3.0
        noise_gain[neg] = 1.0 - (np.abs(breath_raw[neg]) / 100.0) ** 1.2 * 0.85
        noise_gain = np.clip(noise_gain, 0.0, 10.0)
        noise = noise * noise_gain
        print(f"  breath: [{breath_array.min():.1f}, {breath_array.max():.1f}]")

    # voicing — 0=完全消除谐波，100=原始
    if need_voicing:
        voicing_map = interp_to_len(voicing_array, n_samples)
        voicing_map = np.clip(voicing_map, 0, 500)
        # 确保 voicing=0 的采样点谐波彻底归零（避免浮点残留）
        gain = voicing_map / 100.0
        harmonic = harmonic * gain
        # 对 gain≈0 的采样点强制清零
        zero_mask = gain < 1e-8
        if np.any(zero_mask):
            harmonic[zero_mask] = 0.0
        print(f"  voicing: [{voicing_array.min():.1f}, {voicing_array.max():.1f}]")

    # tension
    if need_tension:
        tension_map = interp_to_len(tension_array, n_samples)
        # tension_map = np.clip(tension_map, -100, 100)
        harmonic = apply_dynamic_tension(harmonic, tension_map, sr)
        print(f"  tension: [{tension_array.min():.1f}, {tension_array.max():.1f}]")

    # 混合（不调整整体音量，保持各参数的独立效果）
    wav = noise + harmonic

    print("  HN-SEP 后处理完成")
    return wav
