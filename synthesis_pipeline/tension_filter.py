"""
动态张力（频谱倾斜）滤波器 — STFT 域逐帧处理。

严格参考 hifiserver.py 的 pre_emphasis_base_tension。
使用 numba JIT 加速逐帧循环。
"""
import numpy as np
import librosa

from synthesis_pipeline.utils import interp_to_len

# 尝试导入 numba，不可用时回退到纯 Python
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        """numba 不可用时的占位装饰器。"""
        def decorator(fn):
            return fn
        return decorator


@njit(cache=True, fastmath=True)
def _apply_tilt_to_frames(mag_db: np.ndarray, mag_linear: np.ndarray,
                           tension_frames: np.ndarray,
                           freq_filter_template: np.ndarray,
                           x0: float, n_frames: int, fft_bin: int):
    """在 log 幅度域逐帧应用频谱倾斜滤波器（numba 加速）。

    Args:
        mag_db:       [fft_bin, n_frames] log 幅度谱（原地修改）
        mag_linear:   [fft_bin, n_frames] 原始线性幅度谱（只读）
        tension_frames: [n_frames] 每帧的 tension 值
        freq_filter_template: [fft_bin] 预分配的滤波器数组（会被覆写）
        x0:           1500Hz 对应的 bin 索引
        n_frames:     帧数
        fft_bin:      频点数量

    Returns:
        b_values: [n_frames] 每帧的实际 b 值
    """
    b_values = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        tv = tension_frames[t]
        # 不对称映射：正 tension（提亮）用大除数压住高频刺耳感
        #             负 tension（压暗）保持标准映射
        if tv > 0:
            b = -tv / 150.0   # +100 → b≈-0.67（柔和提亮）
        else:
            b = -tv / 50.0    # -100 → b=+2.0（标准压暗）
        b_values[t] = b

        if abs(b) < 0.001:
            continue  # tension≈0，跳过，该帧完全不变

        # 线性倾斜滤波器: (-b/x0) * bin + b
        for f in range(fft_bin):
            val = (-b / x0) * f + b
            if val > 2.0:
                val = 2.0
            elif val < -2.0:
                val = -2.0
            freq_filter_template[f] = val

        # 原始帧总幅度
        orig_sum = 0.0
        for f in range(fft_bin):
            orig_sum += mag_linear[f, t]

        # 应用滤波器
        for f in range(fft_bin):
            mag_db[f, t] += freq_filter_template[f]

        # 逐帧能量保持：补偿频谱倾斜带来的总幅度变化
        if orig_sum > 1e-12:
            new_sum = 0.0
            for f in range(fft_bin):
                new_sum += np.exp(mag_db[f, t])
            if new_sum > 1e-12:
                comp = np.log(orig_sum / new_sum)
                for f in range(fft_bin):
                    mag_db[f, t] += comp

        # b 相关增益补偿（参考实现，移到逐帧处理）
        # 正 b（压暗）无增益；负 b（提亮）补偿感知响度损失
        b_gain = 0.0
        if b < -0.001:
            val = b / (-15.0)
            if val < 0.0:
                val = 0.0
            elif val > 0.33:
                val = 0.33
            b_gain = np.log(val + 1.0)
        if b_gain > 0.0:
            for f in range(fft_bin):
                mag_db[f, t] += b_gain

    return b_values


def apply_dynamic_tension(waveform: np.ndarray,
                           tension_map: np.ndarray,
                           sr: int = 44100) -> np.ndarray:
    """对音频动态应用张力（频谱倾斜）滤波器。

    参考 hifiserver.py pre_emphasis_base_tension:
      - tension → b: 正 /150（柔和）, 负 /50（标准）
      - 滤波器: (-b/x0) * bin + b, 限制 [-2, 2] dB
      - 逐帧能量保持 + 逐帧 b_gain（无全局归一化，不影响未调区域）
      - 输出仅做溢出软限幅

    Args:
        waveform:    输入音频 (samples,)
        tension_map: 逐样本 tension 值 (-100~100)
        sr:          采样率

    Returns:
        处理后的音频 (samples,)
    """
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    original_len = len(waveform)
    pad_len = (hop_length - (original_len % hop_length)) % hop_length
    padded = np.pad(waveform, (0, pad_len), mode='constant')

    # STFT
    D = librosa.stft(padded, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hann', center=True)
    mag = np.abs(D)
    phase = np.angle(D)

    n_frames = D.shape[1]
    fft_bin = n_fft // 2 + 1

    tension_frames = interp_to_len(tension_map, n_frames)
    x0 = fft_bin / ((sr / 2) / 1500)

    mag_db = np.log(np.clip(mag, 1e-9, None))
    freq_filter_template = np.empty(fft_bin, dtype=np.float32)

    _ = _apply_tilt_to_frames(
        mag_db, mag, tension_frames,
        freq_filter_template, x0, n_frames, fft_bin
    )

    mag_out = np.exp(mag_db)

    # ISTFT
    D_filtered = mag_out * np.exp(1j * phase)
    filtered = librosa.istft(D_filtered, hop_length=hop_length,
                             win_length=win_length, window='hann', center=True)
    filtered = filtered[:original_len]

    # ─── 软限幅（仅在溢出时生效，不改变正常电平） ───
    # 不再做全局峰值归一化（会拖累 tension=0 的区域）。
    # 逐帧能量保持 + 逐帧 b_gain 已在 _apply_tilt_to_frames 中完成。
    peak = np.max(np.abs(filtered))
    if peak > 1.0:
        # tanh 软限幅，避免硬削波
        filtered = np.tanh(filtered * 0.9) / 0.9
        # 缩放回原始峰值附近
        new_peak = np.max(np.abs(filtered))
        if new_peak > 1e-8:
            filtered = filtered * (min(peak, 1.0) / new_peak)

    return filtered
