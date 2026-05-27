"""
HN-SEP ONNX 推理模块

继承 BaseHnsep，差异只在 ONNX 模型加载和推理。
公共逻辑（breath/tension 后处理）在 base_hnsep.py 中。

支持新模型 (pt2.onnx): spec → mask，含外部 STFT/ISTFT
也兼容旧模型: waveform → harmonic + noise
"""
import os as _os
import numpy as np
import onnxruntime
from tools.base_hnsep import BaseHnsep

# STFT 参数（固定，与训练一致）
N_FFT = 2048
HOP_LENGTH = 512
SEG_LENGTH = 32 * HOP_LENGTH  # 16384


def _pad_to_seg(wav: np.ndarray) -> tuple:
    """补齐音频到 seg_length 边界，返回 (padded, tl_pad, original_len)。"""
    T = len(wav)
    T1 = T + HOP_LENGTH
    T_pad = SEG_LENGTH * ((T1 - 1) // SEG_LENGTH + 1) - T1
    nl_pad = T_pad // 2 // HOP_LENGTH
    Tl_pad = nl_pad * HOP_LENGTH
    padded = np.pad(wav, (Tl_pad, T_pad - Tl_pad), mode='constant')
    return padded, Tl_pad, T


def _stft(wav: np.ndarray) -> np.ndarray:
    """numpy STFT，返回 (1, 1, freq, time) 复数。"""
    from scipy import signal
    f, t, Zxx = signal.stft(
        wav, fs=44100, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH,
        window='hann', boundary=None, padded=False
    )
    Zxx = Zxx[np.newaxis, np.newaxis, :, :]  # (1, 1, 1025, T)
    return Zxx.astype(np.complex64)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    """numpy ISTFT，返回时域波形。"""
    from scipy import signal
    _, wav = signal.istft(
        spec[0, 0], fs=44100, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH,
        window='hann', boundary=False
    )
    return wav[:length].astype(np.float32)


class OnnxHnsep(BaseHnsep):
    """HN-SEP 谐波/噪声分离器 — ONNX Runtime 版。"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Args:
            model_path: .onnx 模型路径，默认 pt2 新模型
            device:     'cpu' 或 'dml' 等
        """
        if model_path is None:
            # 优先使用新导出的 pt2 模型
            pt2_path = _os.path.join(
                "hnsep_onnx", "hnsep_VR_44.1k_hop512_2024.05.pt2.onnx")
            if _os.path.exists(pt2_path):
                model_path = pt2_path
            else:
                model_path = _os.path.join(
                    "hnsep_onnx", "hnsep_VR_44.1k_hop512_2024.05.onnx")

        dl = device.lower()
        if dl in ('dml', 'directml'):
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        print(f"加载 HN-SEP ONNX 模型: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self._is_pt2 = 'pt2' in model_path  # 是否新模型格式
        print(f'HN-SEP 加载成功, providers: {self.session.get_providers()}, '
              f'format={"spec->mask" if self._is_pt2 else "waveform->harmonic+noise"}')

    def separate(self, waveform: np.ndarray) -> tuple:
        """分离音频为谐波和噪声分量。"""
        waveform = self._ensure_1d(waveform)

        if self._is_pt2:
            return self._separate_pt2(waveform)
        else:
            return self._separate_legacy(waveform)

    def _separate_legacy(self, waveform: np.ndarray) -> tuple:
        """旧模型：waveform → harmonic + noise。"""
        wav_input = waveform.reshape(1, -1)
        harmonic, noise = self.session.run(
            ['harmonic', 'noise'], {'waveform': wav_input})
        return harmonic[0], noise[0]

    def _separate_pt2(self, waveform: np.ndarray) -> tuple:
        """新模型：外部 STFT → mask → ISTFT → harmonic + noise。"""
        # 1. 补齐 + STFT
        n_samples = len(waveform)
        padded, tl_pad, _ = _pad_to_seg(waveform)
        spec = _stft(padded)  # (1, 1, freq, T), complex64

        # 2. 调用 ONNX 模型
        spec_real = np.ascontiguousarray(spec.real)
        spec_imag = np.ascontiguousarray(spec.imag)
        mask_real, mask_imag = self.session.run(
            ['mask_real', 'mask_imag'],
            {'spec_real': spec_real, 'spec_imag': spec_imag})

        # 3. 复数乘法：spec * mask
        mask = mask_real + 1j * mask_imag
        spec_pred = spec * mask

        # 4. ISTFT → 裁剪补齐
        harmonic = _istft(spec_pred, len(padded))[tl_pad:tl_pad + n_samples]
        noise = waveform[:len(harmonic)] - harmonic

        return harmonic, noise


# ═══════════════════════════════════════════════════════════════
#  全局单例 + 向后兼容函数
# ═══════════════════════════════════════════════════════════════

_global_hnsep_instance: OnnxHnsep | None = None


def get_global_hnsep_session(model_path: str = None, device: str = 'cpu'):
    """获取全局 ONNX HN-SEP 实例（单例）。"""
    global _global_hnsep_instance
    if _global_hnsep_instance is None:
        _global_hnsep_instance = OnnxHnsep(model_path, device)
    return _global_hnsep_instance.session


def preload_hnsep_model(model_path: str = None, device: str = 'cpu'):
    """预加载 HN-SEP ONNX 模型。"""
    try:
        global _global_hnsep_instance
        _global_hnsep_instance = OnnxHnsep(model_path, device)
        print("[OK] HN-SEP 模型预加载成功")
        return True
    except Exception as e:
        print(f"[WARN] HN-SEP 模型预加载失败: {e}")
        return False


def _separate_with_session(waveform: np.ndarray, session) -> tuple:
    """通用分离函数：自动检测新旧模型格式。"""
    wav = BaseHnsep._ensure_1d(waveform)
    # 检查 session 是 pt2 新模型还是旧模型
    try:
        out_names = [o.name for o in session.get_outputs()]
        inp_names = [i.name for i in session.get_inputs()]
    except Exception:
        # 如果是 OnnxHnsep 实例
        return session.separate(waveform)

    if set(out_names) == {'mask_real', 'mask_imag'}:
        # 新模型: spec → mask
        # 用 torch 做 STFT/ISTFT 以保证与训练一致
        import torch
        n_fft, hop_length = 2048, 512
        seg_length = 32 * hop_length

        n_samples = len(wav)
        T1 = n_samples + hop_length
        T_pad = seg_length * ((T1 - 1) // seg_length + 1) - T1
        nl_pad = T_pad // 2 // hop_length
        Tl_pad = nl_pad * hop_length
        padded = np.pad(wav, (Tl_pad, T_pad - Tl_pad), mode='constant')

        wav_t = torch.from_numpy(padded).unsqueeze(0)
        spec = torch.stft(wav_t, n_fft=n_fft, hop_length=hop_length,
                          return_complex=True, window=torch.hann_window(n_fft))
        spec = spec.unsqueeze(0)  # (1, 1, freq, T)

        spec_real = np.ascontiguousarray(spec.real.numpy())
        spec_imag = np.ascontiguousarray(spec.imag.numpy())
        mask_r, mask_i = session.run(
            ['mask_real', 'mask_imag'],
            {'spec_real': spec_real, 'spec_imag': spec_imag})

        mask = torch.from_numpy(mask_r + 1j * mask_i)
        spec_t = torch.view_as_complex(
            torch.stack([spec.real, spec.imag], dim=-1).contiguous())
        spec_pred = spec_t * mask  # (1, 1, 1025, T)

        # ISTFT 需要 (B*C, freq, T) 形状的复数张量
        spec_pred_2d = spec_pred.reshape(1, spec_pred.shape[-2], spec_pred.shape[-1])
        harmonic_t = torch.istft(
            spec_pred_2d, n_fft=n_fft, hop_length=hop_length,
            window=torch.hann_window(n_fft), length=len(padded))
        # istft 输出 (1, padded_len)，取 batch=0，裁剪补齐
        harmonic = harmonic_t[0, Tl_pad:Tl_pad + n_samples].numpy()
        noise = wav[:len(harmonic)] - harmonic
        return harmonic, noise
    else:
        # 旧模型: waveform → harmonic + noise
        wav_input = wav.reshape(1, -1)
        harmonic, noise = session.run(['harmonic', 'noise'], {'waveform': wav_input})
        return harmonic[0], noise[0]


def hnsep_separate(waveform: np.ndarray, session=None) -> tuple:
    """向后兼容：使用指定会话或全局实例分离谐波/噪声。

    支持直接传入 ONNX InferenceSession（post_process.py 的用法）。
    自动检测新旧模型格式。
    """
    if session is not None:
        return _separate_with_session(waveform, session)
    global _global_hnsep_instance
    if _global_hnsep_instance is None:
        _global_hnsep_instance = OnnxHnsep()
    return _global_hnsep_instance.separate(waveform)


def apply_breath_tension(waveform, breath=100, voicing=100, tension=0, session=None):
    """向后兼容：应用 breath/tension 后处理。"""
    global _global_hnsep_instance
    if _global_hnsep_instance is None:
        _global_hnsep_instance = OnnxHnsep()
    return _global_hnsep_instance.apply_breath_tension(
        waveform, breath=breath, voicing=voicing, tension=tension)
    processed = apply_breath_tension(test_wav, breath=150, tension=50)
    print(f"处理后音频: {processed.shape}")

    print("[OK] HN-SEP ONNX 模块测试通过")
