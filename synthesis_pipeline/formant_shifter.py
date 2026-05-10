"""
共振峰偏移模块 — 基于 LPC 的 F1~F4 独立偏移。

原理:
  1. 利用 HN-SEP 将音频分离为谐波分量和噪声分量（可选）
  2. 对谐波（或全频）做 LPC 分析 → 提取极点 → 修改极点频率 → 重合成
  3. 混合回原始噪声分量，最大限度保留气声/摩擦声等非周期成分

用法（在 OpenUTAU Note_flags 中设置）:
  f1=1.15   F1 偏移 1.15 倍（上移 15%）
  f2=0.85   F2 偏移 0.85 倍（下移 15%）
  f3=1.0    F3 不变
  f4=1.0    F4 不变

也可以传递空串或省略，默认 1.0（不变）。
"""

import numpy as np
import scipy.signal as signal
from synthesis_pipeline.utils import interp_to_len

_HAS_LIBROSA = True
try:
    import librosa
except ImportError:
    _HAS_LIBROSA = False

# ─── LPC 工具 ──────────────────────────────────────────────────────────


def lpc_autocorr(x: np.ndarray, order: int) -> np.ndarray:
    """自相关法 LPC 分析（Levinson-Durbin 递归）。

    Args:
        x:     输入信号 (n_samples,)
        order: LPC 阶数

    Returns:
        a: LPC 系数，a[0]=1, a[1..order] 为预测系数，(order+1,)
    """
    n = len(x)
    # 自相关 (biased estimate)
    r = np.correlate(x, x, mode='full')
    r = r[n - 1:n + order].astype(np.float64)

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        rc = -np.dot(a[:i], r[i:0:-1]) / e
        if np.isnan(rc) or np.isinf(rc):
            break
        # 更新 a
        a_new = a.copy()
        half = i // 2
        for j in range(1, half + 1):
            aj = a[j]
            ai_j = a[i - j]
            a_new[j] = aj + rc * ai_j
            a_new[i - j] = ai_j + rc * aj
        if i % 2 == 1:
            mid = half + 1
            a_new[mid] = a[mid] + rc * a[mid]
        a_new[i] = rc
        a = a_new
        e *= (1.0 - rc * rc)
        if e <= 0:
            break

    return a


def _roots_to_formants(roots: np.ndarray, sr: int
                       ) -> list[dict]:
    """从 LPC 多项式根中提取共振峰。

    Args:
        roots: LPC 多项式的根
        sr:    采样率

    Returns:
        list of dict, 每个 dict 含 'freq'(Hz), 'bw'(Hz), 'idx'(roots中的索引),
        按频率升序排列
    """
    formants = []
    for i, r in enumerate(roots):
        if np.imag(r) <= 0:
            continue  # 只取上半平面
        freq = np.abs(np.angle(r)) * sr / (2.0 * np.pi)
        bw = -np.log(np.abs(r)) * sr / np.pi
        if 50 < freq < sr / 2 - 1 and 10 < bw < 2000:
            formants.append({
                'freq': float(freq),
                'bw': float(bw),
                'idx': i,
                'root': r,
            })
    formants.sort(key=lambda f: f['freq'])
    return formants


def _modify_lpc_roots(a: np.ndarray, sr: int,
                      shifts: tuple[float, float, float, float]
                      ) -> np.ndarray:
    """修改 LPC 多项式的根以偏移共振峰。

    Args:
        a:      LPC 系数 (order+1,)
        sr:     采样率
        shifts: (f1_ratio, f2_ratio, f3_ratio, f4_ratio)

    Returns:
        修改后的 LPC 系数 (order+1,)
    """
    order = len(a) - 1
    roots = np.roots(a).astype(np.complex128)

    formants = _roots_to_formants(roots, sr)

    # 取前 4 个共振峰（如有）并应用偏移
    for idx_in_f4, ratio in enumerate(shifts):
        if abs(ratio - 1.0) < 0.001:
            continue
        if idx_in_f4 >= len(formants):
            break
        f = formants[idx_in_f4]
        old_freq = f['freq']
        new_freq = old_freq * ratio
        new_freq = float(np.clip(new_freq, 60, sr / 2 - 10))

        # 修改对应极点的角度（频率）
        r = roots[f['idx']]
        old_theta = np.angle(r)
        new_theta = 2.0 * np.pi * new_freq / sr
        # 保持半径（带宽）不变，只改角度
        new_r = np.abs(r) * np.exp(1j * new_theta)
        roots[f['idx']] = new_r
        # 共轭对
        conj_idx = np.argmin(np.abs(roots - np.conj(r)))
        if conj_idx != f['idx']:
            roots[conj_idx] = np.conj(new_r)

    # 从修改后的根重建多项式
    try:
        new_a = np.poly(roots)
        new_a = np.real(new_a[:order + 1])
        if len(new_a) < order + 1:
            new_a = np.pad(new_a, (order + 1 - len(new_a), 0))
        # 归一化使 a[0] = 1
        new_a = new_a / new_a[0]
    except Exception:
        # 数值不稳定时回退到原始系数
        return a

    # 稳定性检查：所有根必须在单位圆内
    new_roots = np.roots(new_a)
    if np.any(np.abs(new_roots) > 1.0 + 1e-6):
        # 不稳定，回退
        return a

    return new_a


def shift_formants_lpc(
    waveform: np.ndarray,
    sr: int,
    f1_ratio: float = 1.0,
    f2_ratio: float = 1.0,
    f3_ratio: float = 1.0,
    f4_ratio: float = 1.0,
    order: int = 24,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """对音频做 LPC 共振峰偏移（直接波形操作，无 HN-SEP 分离）。

    Args:
        waveform:  输入音频 (samples,)
        sr:        采样率
        f1_ratio:  F1 偏移倍率 (1.0=不变)
        f2_ratio:  F2 偏移倍率
        f3_ratio:  F3 偏移倍率
        f4_ratio:  F4 偏移倍率
        order:     LPC 阶数 (推荐 24 @ 44100Hz)
        frame_ms:  帧长 (毫秒)
        hop_ms:    帧移 (毫秒)

    Returns:
        处理后的音频 (samples,)
    """
    shifts = (f1_ratio, f2_ratio, f3_ratio, f4_ratio)
    if all(abs(r - 1.0) < 0.001 for r in shifts):
        return waveform.copy()

    n_samples = len(waveform)
    frame_len = int(frame_ms * sr / 1000)
    hop_len = int(hop_ms * sr / 1000)

    # 预加重
    pre_emph = 0.97
    wav_p = np.append(waveform[0],
                      waveform[1:] - pre_emph * waveform[:-1])

    # OLA 缓冲区
    output = np.zeros(n_samples, dtype=np.float64)
    window_sum = np.zeros(n_samples, dtype=np.float64)

    n_frames = max(1, (n_samples - frame_len) // hop_len + 1)

    for idx in range(n_frames):
        start = idx * hop_len
        end = start + frame_len
        if end > n_samples:
            start = n_samples - frame_len
            end = n_samples

        frame = wav_p[start:end].astype(np.float64)
        window = np.hanning(frame_len)
        frame_win = frame * window

        # LPC 分析
        a = lpc_autocorr(frame_win, order)

        # 计算残差（激励信号）
        residual = signal.lfilter(a, [1.0], frame_win)

        # 修改共振峰
        a_new = _modify_lpc_roots(a, sr, shifts)

        # 通过新滤波器重合成
        syn = signal.lfilter([1.0], a_new, residual)

        # 去加重
        syn = signal.lfilter([1.0], [1.0, -pre_emph], syn)

        # OLA
        output[start:end] += syn * window
        window_sum[start:end] += window ** 2

    # 归一化
    mask = window_sum > 1e-10
    output[mask] /= window_sum[mask]

    # 能量保持
    orig_rms = np.sqrt(np.mean(waveform.astype(np.float64) ** 2))
    out_rms = np.sqrt(np.mean(output ** 2))
    if out_rms > 1e-10 and orig_rms > 1e-10:
        output *= orig_rms / out_rms

    # 软限幅防削波
    peak = np.max(np.abs(output))
    if peak > 0.95:
        output *= 0.95 / peak

    return output.astype(np.float32)


def shift_formants_with_hnsep(
    waveform: np.ndarray,
    sr: int,
    f1_ratio: float = 1.0,
    f2_ratio: float = 1.0,
    f3_ratio: float = 1.0,
    f4_ratio: float = 1.0,
    hnsep_session=None,
    order: int = 24,
) -> np.ndarray:
    """利用 HN-SEP 分离谐波/噪声后，仅对谐波分量做共振峰偏移。

    这是推荐的方式 — 噪声部分（气声、摩擦声）原封不动，
    共振峰偏移只影响音色本身，音质损失最小。

    Args:
        waveform:     输入音频 (samples,)
        sr:           采样率
        f1_ratio:     F1 偏移倍率
        f2_ratio:     F2 偏移倍率
        f3_ratio:     F3 偏移倍率
        f4_ratio:     F4 偏移倍率
        hnsep_session: HN-SEP ONNX 会话；None 则回退到无分离模式
        order:        LPC 阶数

    Returns:
        处理后的音频 (samples,)
    """
    shifts = (f1_ratio, f2_ratio, f3_ratio, f4_ratio)
    if all(abs(r - 1.0) < 0.001 for r in shifts):
        return waveform.copy()

    if hnsep_session is not None:
        from tools.hnsep_onnx import hnsep_separate
        try:
            harmonic, noise = hnsep_separate(waveform, hnsep_session)

            harmonic_shifted = shift_formants_lpc(
                harmonic, sr, f1_ratio, f2_ratio, f3_ratio, f4_ratio,
                order=order,
            )

            result = harmonic_shifted + noise

            # 能量保持
            orig_rms = np.sqrt(np.mean(waveform.astype(np.float64) ** 2))
            result_rms = np.sqrt(np.mean(result.astype(np.float64) ** 2))
            if result_rms > 1e-10 and orig_rms > 1e-10:
                result = result * (orig_rms / result_rms)

            return result.astype(np.float32)
        except Exception as e:
            print(f"[WARN] HN-SEP 分离失败，回退到全频段共振峰偏移: {e}")
            # fall through

    # 回退：无 HN-SEP 时直接操作全频段
    return shift_formants_lpc(
        waveform, sr, f1_ratio, f2_ratio, f3_ratio, f4_ratio,
        order=order,
    )


# ─── 便捷：从 Note_flags 解析共振峰偏移参数 ──────────────────────────

def parse_formant_flags(flags: dict) -> dict:
    """从 Note_flags 字典中解析共振峰偏移参数。

    Args:
        flags: OpenUTAU Note_flags 字典, 如 {"f1":"1.15","f2":"0.9"}

    Returns:
        {"f1": float, "f2": float, "f3": float, "f4": float}
        未设置的标志默认 1.0
    """
    result = {}
    for key in ('f1', 'f2', 'f3', 'f4'):
        val = flags.get(key)
        if val is not None and val != '':
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                result[key] = 1.0
        else:
            result[key] = 1.0
    return result


def has_formant_shift(parsed: dict) -> bool:
    """检查是否有实际非 1.0 的偏移参数。"""
    return any(abs(v - 1.0) > 0.001 for v in parsed.values())


if __name__ == '__main__':
    # 简单自测
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5))
    # 模拟一个 200Hz 的「a」元音（带前三个谐波）
    test_wav = (0.5 * np.sin(2 * np.pi * 200 * t) +
                0.3 * np.sin(2 * np.pi * 800 * t) +
                0.2 * np.sin(2 * np.pi * 1500 * t) +
                0.1 * np.sin(2 * np.pi * 2500 * t))
    test_wav = test_wav.astype(np.float32)

    # 仿真一声带共振峰的语音（加幅值调制模拟周期脉冲）
    from scipy.signal import lfilter
    # 用一组极点模拟声道
    a_vocal = np.poly([0.98*np.exp(1j*0.1), 0.98*np.exp(-1j*0.1),
                       0.95*np.exp(1j*0.25), 0.95*np.exp(-1j*0.25),
                       0.9*np.exp(1j*0.40), 0.9*np.exp(-1j*0.40),
                       0.85*np.exp(1j*0.55), 0.85*np.exp(-1j*0.55)])
    pulse = np.zeros(int(sr*0.5))
    pulse[::int(sr/200)] = 1.0  # 200Hz 脉冲串
    voiced = lfilter([1.0], a_vocal, pulse)
    voiced = voiced.astype(np.float32)
    voiced /= np.max(np.abs(voiced)) * 0.8

    print("自测: 共振峰偏移...")
    shifted = shift_formants_lpc(voiced, sr, f1_ratio=1.2, f2_ratio=0.85)

    # 简单验证：偏移后能量应基本不变
    orig_rms = np.sqrt(np.mean(voiced ** 2))
    new_rms = np.sqrt(np.mean(shifted ** 2))
    print(f"  原始 RMS: {orig_rms:.6f}")
    print(f"  偏移后 RMS: {new_rms:.6f}")
    print(f"  能量偏差: {abs(orig_rms - new_rms) / orig_rms * 100:.2f}%")
    print("[OK] 自测完成")
