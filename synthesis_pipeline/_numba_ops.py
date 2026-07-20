"""
numba 加速原子操作 — 集中管理所有 @njit 编译的 DSP 核心函数。

各模块通过 from synthesis_pipeline._numba_ops import ... 引用，
避免散落在多个文件中。

设计原则：
  - 每个函数都是纯计算，无 IO、无 print、无副作用
  - 输入 np.ndarray，输出 np.ndarray
  - 全部使用 @njit(cache=True, fastmath=True) 最大化加速
"""

import math
import numpy as np
from numba import njit


# ═══════════════════════════════════════════════════════════════
#  RMS 软膝压缩器
# ═══════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def soft_knee_compressor(
    signal: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float = 6.0,
    attack_ms: float = 5.0,
    release_ms: float = 60.0,
    sr: int = 44100,
) -> np.ndarray:
    """RMS 软膝压缩器（完整 njit 编译，无 Python 回退）。

    与 warmth.py 原 _soft_knee_compressor 功能完全一致，但：
      - RMS 包络用 O(n) 滑动窗口替代 np.convolve
      - 增益曲线用逐样本分支替代 np.where 中间数组
      - IIR 平滑与增益应用合并到一个循环

    Args:
        signal:       输入音频 (samples,)
        threshold_db: 阈值 (dB)
        ratio:        压缩比 (≥1)
        knee_db:      软膝宽度 (dB)
        attack_ms:    启动时间 (ms)
        release_ms:   释放时间 (ms)
        sr:           采样率

    Returns:
        压缩后的音频 (samples,), float32
    """
    n = len(signal)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    x = signal.astype(np.float64)

    # ── RMS 包络（滑动窗口 O(n)，反射边界） ──
    rms_win = max(1, int(sr * 0.01))
    half = rms_win // 2

    # 反射填充 x²
    x2 = x * x
    padded = np.empty(n + 2 * half, dtype=np.float64)
    for i in range(half):
        padded[i] = x2[half - 1 - i]               # 左反射
    for i in range(n):
        padded[half + i] = x2[i]                    # 中间
    for i in range(half):
        padded[half + n + i] = x2[n - 1 - i]        # 右反射

    # 滑动求和
    sum_sq = 0.0
    for i in range(rms_win):
        sum_sq += padded[i]

    inv_win = 1.0 / rms_win
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        rms[i] = math.sqrt(sum_sq * inv_win)
        if i + rms_win < n + 2 * half:
            sum_sq += padded[i + rms_win] - padded[i]

    # ── 增益计算 + IIR 平滑（合并为一个循环） ──
    half_knee = knee_db / 2.0
    one_minus_inv_ratio = 1.0 - 1.0 / ratio

    alpha_a = math.exp(-1.0 / (sr * attack_ms / 1000.0)) if attack_ms > 0 else 0.0
    alpha_r = math.exp(-1.0 / (sr * release_ms / 1000.0)) if release_ms > 0 else 0.0

    out = np.empty(n, dtype=np.float64)
    prev_g = 1.0

    for i in range(n):
        # RMS → dB
        rd = 20.0 * math.log10(max(rms[i], 1e-12))
        os = rd - threshold_db

        # 软膝增益 (dB)
        if os > half_knee:
            g_db = -os * one_minus_inv_ratio
        elif os > -half_knee:
            ko = os + half_knee
            g_db = (ko * ko) / (2.0 * knee_db) * one_minus_inv_ratio
        else:
            g_db = 0.0

        # dB → 线性增益
        g = 10.0 ** (g_db / 20.0)

        # IIR 平滑（attack/release）
        if g < prev_g:
            g_smooth = alpha_a * prev_g + (1.0 - alpha_a) * g
        else:
            g_smooth = alpha_r * prev_g + (1.0 - alpha_r) * g
        prev_g = g_smooth

        # 应用增益
        out[i] = x[i] * g_smooth

    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  电子管式谐波饱和
# ═══════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def apply_saturation(signal: np.ndarray, drive: float) -> np.ndarray:
    """电子管式谐波饱和（完整 njit 编译）。

    与 warmth.py 原 _apply_saturation 功能完全一致，但：
      - 单循环遍历，不产生中间数组 (x_norm, x_driven, x_sat, mixed)
      - 峰值检测与处理合并

    Args:
        signal: 输入音频 (samples,)
        drive:  饱和强度 (0~1)

    Returns:
        饱和处理后的音频 (samples,), float32
    """
    n = len(signal)
    if drive < 0.01 or n == 0:
        return signal.copy().astype(np.float32)

    x = signal.astype(np.float64)

    # 峰值检测
    peak_in = 0.0
    for i in range(n):
        v = abs(x[i])
        if v > peak_in:
            peak_in = v
    if peak_in < 1e-12:
        return signal.copy().astype(np.float32)

    inv_peak = 1.0 / peak_in
    drive_gain = 1.0 + drive * 8.0
    bias = 0.05 * drive
    tanh_bias = math.tanh(bias)
    wet = drive * 0.5
    dry = 1.0 - wet

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        # 归一化 + 驱动
        val = x[i] * inv_peak * drive_gain
        # 两级 tanh（非对称偏置 → 偶次谐波，对称 → 总失真控制）
        val = math.tanh(val + bias) - tanh_bias
        val = math.tanh(val * 1.5)
        # 干湿混合 + 缩放回原始峰值
        out[i] = (dry * (x[i] * inv_peak) + wet * val) * peak_in

    return out.astype(np.float32)
