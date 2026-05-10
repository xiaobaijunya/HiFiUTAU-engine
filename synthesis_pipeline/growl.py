"""
咆哮效果 (Growl) — 在所有后处理之后应用的超快颤音（时域延迟调制）。

通过微小延迟变化弯曲音高，模拟 ~4ms 周期（≈240Hz）的超快颤音，
产生嘶吼/咆哮时声带的快速音高抖动。
growl 值范围 0~100，0=无效果，100=最大深度。
"""
import numpy as np

from synthesis_pipeline.utils import interp_to_len


def apply_growl(waveform: np.ndarray,
                growl_array: np.ndarray,
                sr: int = 44100,
                base_freq: float = 120.0,
                f0: np.ndarray = None,
                f0_hop: int = None) -> np.ndarray:
    """对音频应用动态咆哮效果（时域颤音/延迟调制）。

    使用可变延迟线实现音高调制，延迟信号混合正弦 + 低通噪声，
    打破纯正弦的圆滑感，产生不规则的自然咆哮质感。

    若提供 f0 参数，颤音频率自动跟随音高（freq = max(80, f0 × 0.3)），
    高音快颤、低音慢颤，在不同音域都有最佳表现。

    Args:
        waveform:    输入音频 (samples,)
        growl_array: 逐帧 growl 值 (0~100)，帧间距为 Dynamic_hop
        sr:          采样率
        base_freq:   颤音基准频率 (Hz)，仅在未提供 f0 时生效
        f0:          基频数组 (Hz)，帧间距为 f0_hop，可选
        f0_hop:      f0 的帧间距，与 growl_array 等长时使用 Dynamic_hop

    Returns:
        处理后的音频 (samples,)
    """
    n_samples = len(waveform)

    if len(growl_array) == 0 or np.allclose(growl_array, 0, atol=0.5):
        return waveform

    growl_map = interp_to_len(growl_array, n_samples)
    # growl_map = np.clip(growl_map, 0, 100)

    # ─── 计算每样本的颤音频率 ───
    if f0 is not None and len(f0) > 0 and f0_hop is not None:
        # 跟随音高但限制在咆哮最佳频段 80~150Hz
        f0_map = interp_to_len(f0, n_samples)
        freq = np.clip(f0_map * 0.35, 80.0, 240.0).astype(np.float64)
        # 相位累积器（向量化，处理变频正弦）
        dt = 1.0 / sr
        phase = 2.0 * np.pi * np.cumsum(freq) * dt
        sine = np.sin(phase)
    else:
        freq = np.full(n_samples, base_freq, dtype=np.float64)
        t = np.arange(n_samples, dtype=np.float64) / sr
        sine = np.sin(2.0 * np.pi * base_freq * t)

    # ─── 时域颤音（延迟调制） ───
    # max_delay 必须 < 1/(2π·max_freq) 保证 τ(t) 单调
    max_delay_sec = 0.00012  # 0.12ms，安全上限 1/(2π×150Hz)≈1.06ms

    depth = (growl_map / 100.0) ** 0.8

    # 微量噪声打破正弦圆滑感（比例极低，避免噪音感）
    rng = np.random.RandomState(42)
    raw_noise = rng.randn(n_samples).astype(np.float64)
    # 更激进的低通滤波，截止 ~300Hz
    kernel_size = max(1, int(sr / 300))
    kernel = np.ones(kernel_size) / kernel_size
    noise = np.convolve(raw_noise, kernel, mode='same')
    noise = noise / (np.max(np.abs(noise)) + 1e-8)

    # 正弦主导 + 微量噪声（0.88:0.12）
    modulator = 0.88 * sine + 0.12 * noise
    mod_peak = np.max(np.abs(modulator))
    if mod_peak > 1e-8:
        modulator = modulator / mod_peak

    delay = depth * max_delay_sec * modulator

    # 映射后的时间轴（单调保证，可用 np.interp）
    t = np.arange(n_samples, dtype=np.float64) / sr
    tau = np.clip(t + delay, 0.0, (n_samples - 1) / sr)

    # np.interp 做线性插值（比手写 idx 更平滑）
    x_axis = np.arange(n_samples, dtype=np.float64) / sr
    result = np.interp(tau, x_axis, waveform.astype(np.float64)).astype(np.float32)

    # 轻微峰值限制（插值边界可能产生微幅溢出）
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result * (1.0 / peak)

    avg_freq = float(np.mean(freq))
    print(f"  growl: [{growl_array.min():.1f}, {growl_array.max():.1f}] "
          f"avg_freq={avg_freq:.0f}Hz")

    return result
