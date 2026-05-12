"""
HN-SEP 后处理 — 动态 breath / tension / voicing 参数应用。
"""
import numpy as np

from synthesis_pipeline.utils import interp_to_len
from synthesis_pipeline.tension_filter import apply_dynamic_tension, apply_breath_band_gain


def _hnsep_separate(wav: np.ndarray, hnsep_model) -> tuple:
    """统一分离谐波/噪声，支持 ONNX session 和 PyTorch 模型。"""
    # 如果是 PyTorch 模型（有 separate 方法）
    if hasattr(hnsep_model, 'separate'):
        return hnsep_model.separate(wav)
    # 否则走 ONNX Runtime
    from tools.hnsep_onnx import hnsep_separate as _onnx_sep
    return _onnx_sep(wav, hnsep_model)


def apply_hnsep_postprocess(
    wav: np.ndarray,
    breath_array: np.ndarray,
    tension_array: np.ndarray,
    voicing_array: np.ndarray,
    sr: int,
    hnsep_session,
    f0_curve: np.ndarray = None,
    brel_array: np.ndarray = None,
    breh_array: np.ndarray = None,
) -> np.ndarray:
    """对合成后的音频应用 HN-SEP breath/tension/voicing 后处理。

    Args:
        wav:             合成音频 (samples,)
        breath_array:    Dynamic_hop 帧间距的 breath 值 (-100~100)
        tension_array:   Dynamic_hop 帧间距的 tension 值 (-100~100)
        voicing_array:   Dynamic_hop 帧间距的 voicing 值 (0~500)
        sr:              采样率
        hnsep_session:   HN-SEP ONNX 推理会话 或 PytorchHnsep 实例
        f0_curve:        基频曲线 (Hz)，Dynamic_hop 分辨率，可选
        brel_array:      Dynamic_hop 帧间距的低频气声增益 (-100~100)，可选
        breh_array:      Dynamic_hop 帧间距的高频气声增益 (-100~100)，可选

    Returns:
        处理后的音频
    """
    # 检查是否需要处理
    has_breath = len(breath_array) > 0
    has_tension = len(tension_array) > 0
    has_voicing = len(voicing_array) > 0
    has_brel = len(brel_array) > 0 if brel_array is not None else False
    has_breh = len(breh_array) > 0 if breh_array is not None else False

    if not (has_breath or has_tension or has_voicing or has_brel or has_breh):
        return wav

    need_breath = has_breath and not np.allclose(breath_array, 0, atol=0.5)
    need_tension = has_tension and not np.allclose(tension_array, 0, atol=0.5)
    need_voicing = has_voicing and not np.allclose(voicing_array, 100, rtol=0.05)
    need_brel = has_brel and not np.allclose(brel_array, 0, atol=0.5)
    need_breh = has_breh and not np.allclose(breh_array, 0, atol=0.5)

    if not (need_breath or need_tension or need_voicing or need_brel or need_breh):
        return wav

    print("HN-SEP 后处理: 分离谐波/噪声 + 动态 breath/tension/voicing...")
    harmonic, noise = _hnsep_separate(wav, hnsep_session)
    n_samples = len(wav)

    # breath — 线性增益映射 (-100=静音, 0=原始, +100=×4)
    if need_breath:
        brec_ratio = interp_to_len(breath_array, n_samples) / 100.0
        noise_gain = np.where(brec_ratio > 0,
                              1.0 + brec_ratio * 3.0,      # 正: 1→×4
                              1.0 + brec_ratio)             # 负: 1→0
        noise = noise * np.clip(noise_gain, 0.0, 10.0)
        print(f"  breath: [{breath_array.min():.1f}, {breath_array.max():.1f}]")

    # brel/breh — 低频/高频气声独立增减益
    need_band = need_brel or need_breh
    if need_band:
        # 默认增益 1.0（不变）
        low_gain = np.ones(n_samples, dtype=np.float32)
        high_gain = np.ones(n_samples, dtype=np.float32)

        if need_brel:
            r = interp_to_len(brel_array, n_samples) / 100.0
            low_gain = np.where(r > 0, 1.0 + r * 3.0, 1.0 + r)  # -100=0, 0=1, +100=×4
            print(f"  brel: [{brel_array.min():.1f}, {brel_array.max():.1f}]")
        if need_breh:
            r = interp_to_len(breh_array, n_samples) / 100.0
            high_gain = np.where(r > 0, 1.0 + r * 3.0, 1.0 + r)
            print(f"  breh: [{breh_array.min():.1f}, {breh_array.max():.1f}]")

        noise = apply_breath_band_gain(noise, low_gain, high_gain, sr)

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
        harmonic = apply_dynamic_tension(harmonic, tension_map, sr, f0_curve=f0_curve)
        print(f"  tension: [{tension_array.min():.1f}, {tension_array.max():.1f}]")

    # 混合（不调整整体音量，保持各参数的独立效果）
    wav = noise + harmonic

    print("  HN-SEP 后处理完成")
    return wav
