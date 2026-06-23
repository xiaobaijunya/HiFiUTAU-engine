"""
音频读取 + 通用工具函数（无状态，纯函数）。
"""
import os as _os
import numpy as np
from scipy.interpolate import interp1d
import soundfile as _sf
import librosa as _librosa


def read_audio(loc: str, sr: int | None = None) -> np.ndarray:
    """通用音频读取：支持 WAV/FLAC/OGG/MP3/M4A/AAC 等格式。

    优先使用 soundfile（原生支持 WAV/FLAC/OGG，打包兼容性好），
    对 soundfile 不支持的格式回退到 librosa（需要 ffmpeg 后端）。

    Args:
        loc: 音频文件路径
        sr:  目标采样率。None=保持原始；指定数值则自动重采样

    Returns:
        float32 单声道音频 (samples,)
    """
    if not _os.path.exists(loc):
        raise FileNotFoundError(f"音频文件不存在: {loc}")

    # ── 方案1: soundfile（原生支持 WAV/FLAC/OGG，打包兼容性好）──
    try:
        x, fs = _sf.read(loc, dtype='float32', always_2d=False)
        # 立体声/多声道转单声道
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr is not None and fs != sr:
            x = _librosa.resample(x, orig_sr=fs, target_sr=sr)
        return x.astype(np.float32)
    except Exception:
        pass  # soundfile 不支持该格式（如 MP3/M4A），走 fallback

    # ── 方案2: librosa（支持 MP3/M4A/AAC 等，依赖 ffmpeg）──
    try:
        x, _ = _librosa.load(loc, sr=sr, mono=True)
        return x.astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"无法读取音频文件: {loc}\n"
            f"{type(e).__name__}: {e}\n"
            f"提示: 非 WAV/FLAC/OGG 格式需要安装 ffmpeg"
        )


# ─── 数值工具 ───
def dynamic_range_compression(x: np.ndarray, C: float = 1.0,
                               clip_val: float = 1e-5) -> np.ndarray:
    """mel 谱 log 压缩。"""
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def resample_array(arr: np.ndarray, original_hop: int,
                   target_hop: int) -> np.ndarray:
    """按 hop 比例重采样一维数组（用于 F0 等帧级参数）。"""
    ratio = original_hop / target_hop
    target_len = max(1, round(len(arr) * ratio))
    old_idx = np.arange(len(arr))
    new_idx = np.linspace(0, len(arr) - 1, target_len)
    return interp1d(old_idx, arr, kind='linear',
                    bounds_error=False, fill_value='extrapolate')(new_idx)


def interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    """将一维数组线性插值到目标长度。"""
    if len(arr) == target_len:
        return arr.copy()
    if len(arr) == 0 or target_len <= 0:
        return np.full(target_len,
                       arr[0] if len(arr) > 0 else 0,
                       dtype=np.float32)
    old_idx = np.arange(len(arr))
    new_idx = np.linspace(0, len(arr) - 1, target_len)
    return interp1d(old_idx, arr, kind='linear',
                    bounds_error=False, fill_value='extrapolate')(new_idx)


def hnsep_separate(wav: np.ndarray, hnsep_model):
    """统一分离谐波/噪声，支持 ONNX session 和 PyTorch 模型。

    集中管理，避免各模块重复实现。
    """
    if hasattr(hnsep_model, 'separate'):
        return hnsep_model.separate(wav)
    from tools.hnsep_onnx import hnsep_separate as _onnx_sep
    return _onnx_sep(wav, hnsep_model)
