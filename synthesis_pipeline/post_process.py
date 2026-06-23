"""
HN-SEP 后处理 — 动态 breath / tension / voicing 参数应用。
"""
import numpy as np

from synthesis_pipeline.utils import interp_to_len, hnsep_separate
from synthesis_pipeline.tension_filter import apply_dynamic_tension, apply_breath_band_gain


def apply_hnsep_postprocess_components(
    harmonic: np.ndarray,
    noise: np.ndarray,
    breath_array: np.ndarray,
    tension_array: np.ndarray,
    voicing_array: np.ndarray,
    sr: int,
    f0_curve: np.ndarray = None,
    brel_array: np.ndarray = None,
    breh_array: np.ndarray = None,
) -> tuple:
    """对已分离的谐波/噪声分量应用 breath/tension/voicing 后处理。

    与 apply_hnsep_postprocess 不同，此函数直接操作预先分离好的分量，
    避免重复 HN-SEP 推理。

    Args:
        harmonic:       谐波分量 (samples,)
        noise:          噪声分量 (samples,)
        breath_array:   Dynamic_hop 帧间距的 breath 值 (-100~100)
        tension_array:  Dynamic_hop 帧间距的 tension 值 (-100~100)
        voicing_array:  Dynamic_hop 帧间距的 voicing 值 (0~500)
        sr:             采样率
        f0_curve:       基频曲线 (Hz)，Dynamic_hop 分辨率，可选
        brel_array:     Dynamic_hop 帧间距的低频气声增益 (-100~100)，可选
        breh_array:     Dynamic_hop 帧间距的高频气声增益 (-100~100)，可选

    Returns:
        (harmonic, noise) 处理后的谐波和噪声分量
    """
    n_samples = len(harmonic)

    # breath — 线性增益映射 (-100=静音, 0=原始, +100=×4)
    if len(breath_array) > 0 and not np.allclose(breath_array, 0, atol=0.5):
        brec_ratio = interp_to_len(breath_array, n_samples) / 100.0
        noise_gain = np.where(brec_ratio > 0,
                              1.0 + brec_ratio * 3.0,      # 正: 1→×4
                              1.0 + brec_ratio)             # 负: 1→0
        noise = noise * np.clip(noise_gain, 0.0, 10.0)
        print(f"  breath: [{breath_array.min():.1f}, {breath_array.max():.1f}]")

    # brel/breh — 低频/高频气声独立增减益
    has_brel = len(brel_array) > 0 if brel_array is not None else False
    has_breh = len(breh_array) > 0 if breh_array is not None else False
    need_brel = has_brel and not np.allclose(brel_array, 0, atol=0.5)
    need_breh = has_breh and not np.allclose(breh_array, 0, atol=0.5)
    need_band = need_brel or need_breh
    if need_band:
        low_gain = np.ones(n_samples, dtype=np.float32)
        high_gain = np.ones(n_samples, dtype=np.float32)
        if need_brel:
            r = interp_to_len(brel_array, n_samples) / 100.0
            low_gain = np.where(r > 0, 1.0 + r * 3.0, 1.0 + r)
            print(f"  brel: [{brel_array.min():.1f}, {brel_array.max():.1f}]")
        if need_breh:
            r = interp_to_len(breh_array, n_samples) / 100.0
            high_gain = np.where(r > 0, 1.0 + r * 3.0, 1.0 + r)
            print(f"  breh: [{breh_array.min():.1f}, {breh_array.max():.1f}]")
        noise = apply_breath_band_gain(noise, low_gain, high_gain, sr)

    # voicing — 0=完全消除谐波，100=原始
    if len(voicing_array) > 0 and not np.allclose(voicing_array, 100, rtol=0.05):
        voicing_map = interp_to_len(voicing_array, n_samples)
        voicing_map = np.clip(voicing_map, 0, 500)
        gain = voicing_map / 100.0
        harmonic = harmonic * gain
        zero_mask = gain < 1e-8
        if np.any(zero_mask):
            harmonic[zero_mask] = 0.0
        print(f"  voicing: [{voicing_array.min():.1f}, {voicing_array.max():.1f}]")

    # tension
    if len(tension_array) > 0 and not np.allclose(tension_array, 0, atol=0.5):
        tension_map = interp_to_len(tension_array, n_samples)
        harmonic = apply_dynamic_tension(harmonic, tension_map, sr, f0_curve=f0_curve)
        print(f"  tension: [{tension_array.min():.1f}, {tension_array.max():.1f}]")

    return harmonic, noise


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
    """向后兼容包装器：分离 → 处理分量 → 混合。

    新代码请直接使用 apply_hnsep_postprocess_components 配合预先分离的分量。
    """
    print("HN-SEP 后处理: 分离谐波/噪声 + 动态 breath/tension/voicing...")
    harmonic, noise = hnsep_separate(wav, hnsep_session)
    harmonic, noise = apply_hnsep_postprocess_components(
        harmonic, noise, breath_array, tension_array, voicing_array, sr,
        f0_curve=f0_curve, brel_array=brel_array, breh_array=breh_array,
    )
    wav = noise + harmonic
    print("  HN-SEP 后处理完成")
    return wav
