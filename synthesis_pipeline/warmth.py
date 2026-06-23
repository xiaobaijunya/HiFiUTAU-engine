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


def _soft_knee_compressor(signal: np.ndarray,
                          threshold_db: float,
                          ratio: float,
                          knee_db: float = 6.0,
                          attack_ms: float = 5.0,
                          release_ms: float = 60.0,
                          sr: int = 44100) -> np.ndarray:
    """RMS 软膝压缩器（向量化版本）。

    用卷积代替 Python 循环计算 RMS 包络，np.where 向量化增益曲线。

    Args:
        signal:       输入音频 (samples,)
        threshold_db: 阈值 (dB)
        ratio:        压缩比 (≥1)
        knee_db:      软膝宽度 (dB)
        attack_ms:    启动时间 (ms)
        release_ms:   释放时间 (ms)
        sr:           采样率

    Returns:
        压缩后的音频 (samples,)
    """
    n = len(signal)
    if n == 0:
        return signal

    x = signal.astype(np.float64)

    # ── RMS 包络（卷积法，O(n) 向量化） ──
    rms_win = max(1, int(sr * 0.01))
    kernel = np.ones(rms_win, dtype=np.float64) / rms_win
    # 边界处理：使用 'same' 模式 + 边界扩展保证 RMS 窗口完整
    x_pad = np.pad(x ** 2, rms_win // 2, mode='reflect')
    rms_sq = np.convolve(x_pad, kernel, mode='same')
    rms = np.sqrt(np.clip(rms_sq[rms_win // 2:rms_win // 2 + n], 0.0, None))

    rms_db = 20.0 * np.log10(np.clip(rms, 1e-12, None))

    # ── 软膝增益曲线（向量化 np.where） ──
    overshoot = rms_db - threshold_db
    half_knee = knee_db / 2.0

    mask_low = overshoot < -half_knee          # 不压缩
    mask_high = overshoot > half_knee           # 硬压缩
    mask_knee = ~(mask_low | mask_high)         # 软膝过渡

    gain_db = np.zeros(n, dtype=np.float64)
    # 硬压缩区
    gain_db[mask_high] = -overshoot[mask_high] * (1.0 - 1.0 / ratio)
    # 软膝过渡区
    knee_os = overshoot[mask_knee] + half_knee
    gain_db[mask_knee] = (knee_os ** 2) / (2.0 * knee_db) * (1.0 - 1.0 / ratio)
    # mask_low 保持 0

    # ── attack/release 平滑（一阶 IIR，时序依赖，保留逐样本循环） ──
    gain_linear = 10.0 ** (gain_db / 20.0)

    alpha_a = np.exp(-1.0 / (sr * attack_ms / 1000.0)) if attack_ms > 0 else 0.0
    alpha_r = np.exp(-1.0 / (sr * release_ms / 1000.0)) if release_ms > 0 else 0.0

    smoothed = np.empty(n, dtype=np.float64)
    smoothed[0] = 1.0
    for i in range(1, n):
        t = gain_linear[i]
        if t < smoothed[i - 1]:
            smoothed[i] = alpha_a * smoothed[i - 1] + (1.0 - alpha_a) * t
        else:
            smoothed[i] = alpha_r * smoothed[i - 1] + (1.0 - alpha_r) * t

    return (x * smoothed).astype(np.float32)


def _apply_saturation(signal: np.ndarray,
                      drive: float,
                      sr: int = 44100) -> np.ndarray:
    """电子管式谐波饱和。

    使用 soft-clipping + 偶次谐波增强，模拟电子管温暖染色。
    drive 范围 0~1。

    Args:
        signal: 输入音频 (samples,)
        drive:  饱和强度 (0~1)
        sr:     采样率

    Returns:
        饱和处理后的音频 (samples,)
    """
    if drive < 0.01:
        return signal

    x = signal.astype(np.float64)

    # 归一化到峰值，饱和后再缩放回来
    peak_in = np.max(np.abs(x))
    if peak_in < 1e-12:
        return signal

    x_norm = x / peak_in

    # 级联 tanh 模拟电子管饱和
    # drive 控制驱动强度
    drive_gain = 1.0 + drive * 8.0  # 1x ~ 9x
    x_driven = x_norm * drive_gain

    # 两级 tanh 产生偶次谐波
    # 第一级: 非对称轻微偏置 → 引入偶次谐波
    bias = 0.05 * drive
    x_sat = np.tanh(x_driven + bias) - np.tanh(bias)
    # 第二级: 对称饱和 → 控制总失真量
    x_sat = np.tanh(x_sat * 1.5)

    # 干湿混合
    wet = drive * 0.5  # 湿信号最大混合比 50%
    mixed = (1.0 - wet) * x_norm + wet * x_sat

    # 缩放回原始峰值（保持响度不变）
    result = mixed * peak_in
    return result.astype(np.float32)


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
        x = _soft_knee_compressor(x, threshold_db=threshold, ratio=ratio, sr=sr)

        # 2. 管饱和：增加偶次谐波温暖染色（效果减半）
        x = _apply_saturation(x, drive=warmth * 0.3, sr=sr)

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
        x = _soft_knee_compressor(x, threshold_db=threshold, ratio=ratio, sr=sr)

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
    x = _soft_knee_compressor(x, threshold_db=threshold, ratio=ratio,
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
