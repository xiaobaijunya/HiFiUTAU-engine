"""
温暖度处理器 — 模拟风格的人声温暖压缩 + 谐波饱和。

模拟模拟调音台的温暖通道条效果，包含：
  1. 软膝压缩（RMS 检波）— 压低峰值、提升细节，让声音更「稳定」
  2. 电子管式谐波饱和 — 增加偶次谐波，产生「温暖」的听感
  3. 温和的 EQ 塑形 — 提升低频段 (200~400Hz)，让人声更「突出」
  4. 能量归一化 — 输出音量与输入一致

warmth > 0：温暖（压缩 + 饱和 + 低频提升）
warmth < 0：清凉（反向 — 扩展 + 高频提升），使声音偏亮偏薄
warmth = 0：直通
"""

import numpy as np
import librosa

from synthesis_pipeline.utils import hnsep_separate
from synthesis_pipeline._numba_ops import (
    soft_knee_compressor,
    apply_saturation,
)


# _soft_knee_compressor 已移至 _numba_ops.py 中的 soft_knee_compressor


# _apply_saturation 已移至 _numba_ops.py 中的 apply_saturation


def _apply_stft_eq(x: np.ndarray, fc: float, gain_db: float,
                    sigma_oct: float = 2.0, sr: int = 44100) -> np.ndarray:
    """STFT 域钟形 EQ 辅助函数，减少代码重复。"""
    n_fft = 2048
    hop_length = 512
    original_len = len(x)
    pad_len = (hop_length - (original_len % hop_length)) % hop_length
    padded = np.pad(x, (0, pad_len), mode='constant')
    D = librosa.stft(padded, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
    mag = np.abs(D)
    phase = np.angle(D)
    fft_bin = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, fft_bin)
    log_f = np.zeros_like(freqs)
    mask_pos = freqs > 1.0
    log_f[mask_pos] = np.log2(freqs[mask_pos])
    log_f[~mask_pos] = log_f[mask_pos][0] if np.any(mask_pos) else -10.0
    bell = np.exp(-0.5 * ((log_f - np.log2(fc)) / sigma_oct) ** 2)
    curve = 2.0 * bell - 1.0
    neg_mask = curve < 0
    if np.any(neg_mask):
        curve[neg_mask] = -(-curve[neg_mask]) ** 0.7
    gain_curve = curve * gain_db
    mag_db = np.log(np.clip(mag, 1e-12, None))
    mag_db += gain_curve[:, np.newaxis]
    D_filtered = np.exp(mag_db) * np.exp(1j * phase)
    out = librosa.istft(D_filtered, hop_length=hop_length, window='hann', center=True)
    return out[:original_len]


def apply_warmth_eq(waveform: np.ndarray,
                    warmth_value: float,
                    sr: int = 44100,
                    hnsep_session=None) -> np.ndarray:
    """对音频应用温暖度处理器（压缩 + 饱和 + EQ），仅作用于谐波分量。

    模拟模拟调音台的温暖通道条：
    - warmth > 0（正值）：温暖 — 压缩动态 + 管饱和 + 低频提升
    - warmth < 0（负值）：清凉 — 轻压缩 + 中高频提升
    - warmth = 0：直通

    若有 hnsep_session，先分离谐波/噪声，只对谐波部分处理，
    避免气噪（噪声）被压缩/饱和放大导致音质劣化。

    Args:
        waveform:     输入音频 (samples,)
        warmth_value: 温暖度值 (-100~100)
        sr:           采样率
        hnsep_session: HN-SEP 模型（可选），用于分离谐波/噪声

    Returns:
        处理后的音频 (samples,)
    """
    # 反转符号：OpenUTAU 发送 warm=-100 表示温暖，内部用正值处理
    warmth_value = -warmth_value

    if abs(warmth_value) < 0.5:
        return waveform

    warmth = warmth_value / 100.0  # [-1, 1]
    x = waveform.astype(np.float64)

    # ── 分离谐波/噪声（如果有 HN-SEP） ──
    noise = None
    if hnsep_session is not None:
        try:
            harmonic, noise = hnsep_separate(waveform, hnsep_session)
            harmonic = harmonic.astype(np.float64)
            noise = noise.astype(np.float64)
            x = harmonic  # 只处理谐波部分
            print(f"  温暖度: 分离谐波/噪声，仅处理谐波 ({len(x)} 采样)")
        except Exception:
            noise = None

    # ─── warmth > 0：温暖模式（压缩 + 饱和 + 低频提升） ───
    if warmth > 0:
        # 1. 压缩：降低动态范围，让声音更「稳定」
        peak_db = 20.0 * np.log10(np.max(np.abs(x)) + 1e-12)
        threshold = peak_db - 18.0 + (1.0 - warmth) * 6.0
        ratio = 1.0 + warmth * 3.0  # warmth=1 → 4:1
        x = soft_knee_compressor(x, threshold_db=threshold, ratio=ratio, sr=sr)

        # 2. 管饱和：增加偶次谐波温暖染色（效果减半）
        x = apply_saturation(x, drive=warmth * 0.3)

        # 3. EQ：宽带提升 300Hz 温暖频段（效果减半）
        x = _apply_stft_eq(x, fc=300.0, gain_db=warmth * 1.25, sigma_oct=2.4, sr=sr)

    # ─── warmth < 0：清凉模式（轻压缩 + 中高频提升） ───
    else:
        cool = -warmth  # [0, 1]

        # 1. 轻压缩（不是扩展！）：保持动态稳定，避免音量异常
        #    用更温和的压缩，ratio 最高 2:1，维持声音自然
        peak_db = 20.0 * np.log10(np.max(np.abs(x)) + 1e-12)
        threshold = peak_db - 20.0
        ratio = 1.0 + cool * 1.0  # cool=1 → 2:1
        x = soft_knee_compressor(x, threshold_db=threshold, ratio=ratio, sr=sr)

        # 2. EQ：提升 5kHz 明亮频段
        x = _apply_stft_eq(x, fc=5000.0, gain_db=cool * 2.5, sigma_oct=3.0, sr=sr)

    # ── 若有 HN-SEP 分离，将处理后的谐波与原始噪声混合 ──
    if noise is not None:
        x = x + noise

    # ── 能量归一化 ──
    rms_in = np.sqrt(np.mean(waveform.astype(np.float64) ** 2))
    rms_out = np.sqrt(np.mean(x ** 2))
    if rms_out > 1e-12 and rms_in > 1e-12:
        gain = rms_in / rms_out
        gain = np.clip(gain, 0.01, 100.0)
        x = x * gain

    # ── 软限幅 ──
    peak = np.max(np.abs(x))
    if peak > 0.99:
        x = x * (0.99 / peak)

    return x.astype(np.float32)


def apply_harmonic_compression(waveform: np.ndarray,
                                hcmp_value: float,
                                sr: int = 44100,
                                hnsep_session=None) -> np.ndarray:
    """对谐波分量施加 RMS 压缩，让谐波音量更「稳定」。

    与温暖度不同，此函数不做饱和和 EQ，只做纯压缩。
    若有 HN-SEP 则仅压缩谐波部分，气噪不受影响。

    Args:
        waveform:     输入音频 (samples,)
        hcmp_value:   压缩强度 (0~100)，0=无效果
        sr:           采样率
        hnsep_session: HN-SEP 模型（可选）

    Returns:
        处理后的音频 (samples,)
    """
    if abs(hcmp_value) < 0.5:
        return waveform

    hcmp = hcmp_value / 100.0  # [0, 1]
    x = waveform.astype(np.float64)

    # ── 分离谐波/噪声 ──
    noise = None
    if hnsep_session is not None:
        try:
            harmonic, noise = hnsep_separate(waveform, hnsep_session)
            harmonic = harmonic.astype(np.float64)
            noise = noise.astype(np.float64)
            x = harmonic
        except Exception:
            noise = None

    # ── 软膝压缩（效果翻倍） ──
    peak_db = 20.0 * np.log10(np.max(np.abs(x)) + 1e-12)
    # 固定阈值（-22dB 相对峰值），HCMP 仅控制压缩比
    threshold = peak_db - 22.0
    # hcmp=0 → 1:1（无压缩），hcmp=100 → 20:1（强压缩）
    ratio = 1.0 + hcmp * 19.0
    x = soft_knee_compressor(x, threshold_db=threshold, ratio=ratio,
                              attack_ms=1.0, sr=sr)

    # ── 混合噪声 ──
    if noise is not None:
        x = x + noise

    # ── 能量归一化 ──
    rms_in = np.sqrt(np.mean(waveform.astype(np.float64) ** 2))
    rms_out = np.sqrt(np.mean(x ** 2))
    if rms_out > 1e-12 and rms_in > 1e-12:
        gain = rms_in / rms_out
        gain = np.clip(gain, 0.01, 100.0)
        x = x * gain

    # ── 软限幅 ──
    peak = np.max(np.abs(x))
    if peak > 0.99:
        x = x * (0.99 / peak)

    return x.astype(np.float32)
