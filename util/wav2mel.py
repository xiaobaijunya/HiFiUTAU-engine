"""
可调 mel 频谱提取器（NumPy 实现）

支持自定义帧中心位置（用于时间拉伸）和频域偏移（用于移调）。
"""
import numpy as np
from typing import Optional
from librosa.filters import mel as librosa_mel_fn


def centered_stft(
    x: np.ndarray,
    centers: np.ndarray,
    n_fft: int,
    win_length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
    pad_mode: str = 'reflect',
    normalized: bool = False,
    onesided: bool = True,
    return_complex: bool = True
) -> np.ndarray:
    """指定中心位置的 STFT（NumPy 实现）。"""
    is_unbatched = (x.ndim == 1)
    if is_unbatched:
        x = x[np.newaxis, :]

    if x.ndim != 2:
        raise ValueError(f"输入 x 维度应为 1 或 2，得到 {x.ndim}")

    if win_length is None:
        win_length = n_fft

    if window is None:
        window = np.hanning(win_length).astype(x.dtype)
    else:
        window = window.astype(x.dtype)

    if window.shape[0] != n_fft:
        pad_left = (n_fft - window.shape[0]) // 2
        pad_right = n_fft - window.shape[0] - pad_left
        window = np.pad(window, (pad_left, pad_right), mode='constant', constant_values=0)

    pad_amount = n_fft // 2
    pad_width = ((0, 0), (pad_amount, pad_amount)) if x.ndim == 2 else (pad_amount, pad_amount)
    x_padded = np.pad(x, pad_width, mode=pad_mode)

    start_indices = centers + pad_amount - (n_fft // 2)
    offset = np.arange(n_fft)
    indices = start_indices[:, np.newaxis] + offset[np.newaxis, :]
    indices = np.clip(indices.astype(int), 0, x_padded.shape[-1] - 1)

    frames = x_padded[:, indices] * window

    if onesided:
        stft = np.fft.rfft(frames, n=n_fft, axis=-1)
    else:
        stft = np.fft.fft(frames, n=n_fft, axis=-1)

    if normalized:
        stft = stft / np.sqrt(n_fft)

    if not return_complex:
        stft = np.stack((stft.real, stft.imag), axis=-1)

    if is_unbatched:
        stft = stft[0]
    return stft


class PitchAndTimeAdjustableMelSpectrogram:
    """支持自定义帧位置和移调的 Mel 频谱提取器。"""

    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        f_min=40,
        f_max=16000,
        n_mels=128,
        mel_fn=librosa_mel_fn,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

        mel = mel_fn(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        self.mel_basis = np.asarray(mel, dtype=np.float64)
        self.hann_window = {}

    def __call__(self, y: np.ndarray, centers: np.ndarray, key_shift: int = 0) -> np.ndarray:
        """
        Args:
            y: 音频信号 (batch, samples) 或 (samples,)
            centers: 中心帧索引 (n_frames,)
            key_shift: 半音移调量
        Returns:
            Mel 频谱图 (batch, n_mels, n_frames) 或 (n_mels, n_frames)
        """
        is_unbatched = (y.ndim == 1)
        if is_unbatched:
            y = y[np.newaxis, :]

        factor = 2.0 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))

        if key_shift not in self.hann_window:
            self.hann_window[key_shift] = np.hanning(win_size_new).astype(y.dtype)

        spec = centered_stft(
            y, centers, n_fft_new,
            win_length=win_size_new,
            window=self.hann_window[key_shift],
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = np.abs(spec)
        spec = spec.transpose(0, 2, 1)

        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.shape[1]
            if resize < size:
                pad_width = size - resize
                spec = np.pad(spec, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            spec = spec[:, :size, :]
            spec = spec * (self.win_size / win_size_new)

        mel_spec = self.mel_basis @ spec

        if is_unbatched:
            mel_spec = mel_spec[0]

        return mel_spec
