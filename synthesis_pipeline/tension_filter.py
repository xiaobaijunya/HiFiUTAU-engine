"""
动态张力（频谱倾斜）滤波器 — STFT 域逐帧处理。

严格参考 hifiserver.py 的 pre_emphasis_base_tension。
使用 numba JIT 加速逐帧循环。
"""
from typing import Optional

import numpy as np
import librosa
from scipy import signal as _signal

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
                           x0_per_frame: np.ndarray,
                           n_frames: int, fft_bin: int):
    """在 log 幅度域逐帧应用频谱倾斜滤波器（numba 加速）。

    Args:
        mag_db:       [fft_bin, n_frames] log 幅度谱（原地修改）
        mag_linear:   [fft_bin, n_frames] 原始线性幅度谱（只读）
        tension_frames: [n_frames] 每帧的 tension 值
        freq_filter_template: [fft_bin] 预分配的滤波器数组（会被覆写）
        x0_per_frame: [n_frames] 每帧的 0-crossing bin 索引（跟随音高）
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

        x0 = x0_per_frame[t]
        if x0 <= 0:
            x0 = 1.0  # 防除零

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
                           sr: int = 44100,
                           f0_curve: Optional[np.ndarray] = None) -> np.ndarray:
    """对音频动态应用张力（频谱倾斜）滤波器。

    参考 hifiserver.py pre_emphasis_base_tension:
      - tension → b: 正 /150（柔和）, 负 /50（标准）
      - 滤波器: (-b/x0) * bin + b, 限制 [-2, 2] dB
      - 逐帧能量保持 + 逐帧 b_gain（无全局归一化，不影响未调区域）
      - 输出仅做溢出软限幅
      - 若提供 f0_curve，中点自动跟随第 4 谐波（f0 × 4），否则固定 1500Hz

    Args:
        waveform:    输入音频 (samples,)
        tension_map: 逐样本 tension 值 (-100~100)
        sr:          采样率
        f0_curve:    基频曲线 (Hz)，任意分辨率，可选。提供后中点 = f0 × 4

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

    # 每帧的 0-crossing bin 索引：跟随音高或固定 1500Hz
    if f0_curve is not None and len(f0_curve) > 0:
        f0_frames = interp_to_len(f0_curve, n_frames)
        # 中点 = 第 4 谐波，限制合理范围
        midpoint_hz = np.clip(f0_frames * 4.0, 400.0, 6000.0)
        # 无声/清音区（f0≈0）回退到 1500Hz
        silent = f0_frames < 30.0
        midpoint_hz[silent] = 1500.0
        x0_per_frame = fft_bin / ((sr / 2) / midpoint_hz)
        x0_per_frame = x0_per_frame.astype(np.float32)
    else:
        x0_fixed = fft_bin / ((sr / 2) / 1500)
        x0_per_frame = np.full(n_frames, x0_fixed, dtype=np.float32)

    mag_db = np.log(np.clip(mag, 1e-9, None))
    freq_filter_template = np.empty(fft_bin, dtype=np.float32)

    _ = _apply_tilt_to_frames(
        mag_db, mag, tension_frames,
        freq_filter_template, x0_per_frame, n_frames, fft_bin
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


# ═══════════════════════════════════════════════════════════════
#  低切（动态高通）— STFT 域 Butterworth 响应，跟随 F0
# ═══════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _apply_lowcut_to_frames(mag_db: np.ndarray,
                             x0_per_frame: np.ndarray,
                             n_frames: int, fft_bin: int):
    """在 log 幅度域逐帧应用 Butterworth 高通滤波器（numba 加速）。

    2 阶 Butterworth 响应: gain_db = -10 * log10(1 + (fc/f)^4)
    12dB/oct 平缓斜率，无能量补偿（低切就是要去能量）。

    Args:
        mag_db:       [fft_bin, n_frames] log 幅度谱（原地修改）
        x0_per_frame: [n_frames] 每帧的截止频率对应 bin 索引
        n_frames:     帧数
        fft_bin:      频点数量
    """
    for t in range(n_frames):
        cutoff_bin = x0_per_frame[t]
        if cutoff_bin <= 0.5:
            continue  # 截止≈0，跳过该帧

        for f in range(fft_bin):
            if f == 0:
                mag_db[0, t] -= 80.0  # DC 彻底切掉
                continue
            # Butterworth 2nd order high-pass: |H|² = 1/(1+(fc/f)^4)
            ratio = cutoff_bin / f
            r2 = ratio * ratio
            gain_db = -10.0 * np.log10(1.0 + r2 * r2)
            mag_db[f, t] += gain_db


def apply_dynamic_lowcut(waveform: np.ndarray,
                          lowcut_map: np.ndarray,
                          sr: int = 44100,
                          f0_curve: Optional[np.ndarray] = None) -> np.ndarray:
    """对音频动态应用 F0 跟随低切滤波器（STFT 域 Butterworth 高通）。

    设计：
      - Butterworth 2nd order 响应（12dB/oct 平缓斜率）
      - 截止频率 = 20 + (lowcut/100) × F0 × 0.6（Hz）
      - 无声/清音区回退到最小截止 20Hz
      - 无能量补偿（低切就是要去掉多余低频）

    Args:
        waveform:    输入音频 (samples,)
        lowcut_map: 逐样本低切值 (0~100)，0=无效果
        sr:          采样率
        f0_curve:    基频曲线 (Hz)，可选。提供后截止频率跟随 F0

    Returns:
        处理后的音频 (samples,)
    """
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    original_len = len(waveform)
    pad_len = (hop_length - (original_len % hop_length)) % hop_length
    padded = np.pad(waveform, (0, pad_len), mode='constant')

    D = librosa.stft(padded, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hann', center=True)
    mag = np.abs(D)
    phase = np.angle(D)

    n_frames = D.shape[1]
    fft_bin = n_fft // 2 + 1

    lowcut_frames = interp_to_len(lowcut_map, n_frames)

    # 快速跳过：参数全为 0 则不处理
    if np.max(lowcut_map) < 0.5:
        return waveform

    # 每帧的截止频率（bin 索引）
    if f0_curve is not None and len(f0_curve) > 0:
        f0_frames = interp_to_len(f0_curve, n_frames)
        # cutoff = (lowcut/100) × F0，0~100% 逐渐逼近 F0
        cutoff_hz = (lowcut_frames / 100.0) * f0_frames
        # 无声/清音区（f0≈0）不切
        cutoff_hz[f0_frames < 30.0] = 0.0
        # 上限保护，防止极高音过量
        cutoff_hz = np.clip(cutoff_hz, 0.0, 2000.0)
    else:
        # 无 F0 时用固定值
        cutoff_hz = lowcut_frames * 5.0
        cutoff_hz = np.clip(cutoff_hz, 0.0, 500.0)

    x0_per_frame = np.zeros(n_frames, dtype=np.float32)
    mask = cutoff_hz > 1.0
    x0_per_frame[mask] = fft_bin / ((sr / 2) / cutoff_hz[mask])

    mag_db = np.log(np.clip(mag, 1e-9, None))

    _apply_lowcut_to_frames(
        mag_db, x0_per_frame, n_frames, fft_bin
    )

    mag_out = np.exp(mag_db)

    D_filtered = mag_out * np.exp(1j * phase)
    filtered = librosa.istft(D_filtered, hop_length=hop_length,
                             win_length=win_length, window='hann', center=True)
    filtered = filtered[:original_len]

    return filtered


# ── 缓存 Butterworth 分频器系数（crossover=2000Hz, sr=44100） ──
# 避免 apply_breath_band_gain 每次调用重新计算滤波器
_BAND_SOS_LOW = _signal.butter(4, 2000.0, btype='low', fs=44100, output='sos')
_BAND_SOS_HIGH = _signal.butter(4, 2000.0, btype='high', fs=44100, output='sos')


def apply_breath_band_gain(waveform: np.ndarray,
                            low_gain: np.ndarray,
                            high_gain: np.ndarray,
                            sr: int = 44100,
                            crossover_hz: float = 2000.0) -> np.ndarray:
    """对气声（噪声）分频后独立乘线性增益，再相加。

    用 Butterworth 滤波器分离低频和高频，各自乘以线性增益系数，再加起来。
    和调音台推子一样直白。

    Args:
        waveform:    输入音频 (samples,)
        low_gain:    低频段逐样本线性增益
        high_gain:   高频段逐样本线性增益
        sr:          采样率
        crossover_hz: 分频点频率 (Hz)

    Returns:
        处理后的音频 (samples,)
    """
    n = len(waveform)
    if n == 0:
        return waveform
    if np.max(np.abs(low_gain - 1.0)) < 0.01 and np.max(np.abs(high_gain - 1.0)) < 0.01:
        return waveform

    # 使用缓存的滤波器系数（默认 2000Hz/44100），非标准参数时才重新计算
    if abs(crossover_hz - 2000.0) < 0.1 and sr == 44100:
        sos_low, sos_high = _BAND_SOS_LOW, _BAND_SOS_HIGH
    else:
        sos_low = _signal.butter(4, crossover_hz, btype='low', fs=sr, output='sos')
        sos_high = _signal.butter(4, crossover_hz, btype='high', fs=sr, output='sos')

    low = _signal.sosfilt(sos_low, waveform)
    high = _signal.sosfilt(sos_high, waveform)

    return low * low_gain + high * high_gain

