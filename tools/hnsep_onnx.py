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
        """新模型：numpy STFT → mask → numpy ISTFT（无 torch/scipy 依赖）。"""
        n_fft, hop_length = 2048, 512
        seg_length = 32 * hop_length
        window = np.hanning(n_fft).astype(np.float32)

        n_samples = len(waveform)
        T1 = n_samples + hop_length
        T_pad = seg_length * ((T1 - 1) // seg_length + 1) - T1
        nl_pad = T_pad // 2 // hop_length
        Tl_pad = nl_pad * hop_length
        padded = np.pad(waveform, (Tl_pad, T_pad - Tl_pad), mode='constant')

        # numpy STFT
        pad_len = n_fft // 2
        buf = np.pad(padded, (pad_len, pad_len), mode='reflect')
        n_frames = (len(buf) - n_fft) // hop_length + 1
        spec = np.zeros((1, 1, n_fft // 2 + 1, n_frames), dtype=np.complex64)
        for t in range(n_frames):
            idx = t * hop_length
            spec[0, 0, :, t] = np.fft.rfft(buf[idx:idx + n_fft] * window)

        mask_r, mask_i = self.session.run(
            ['mask_real', 'mask_imag'],
            {'spec_real': np.ascontiguousarray(spec.real),
             'spec_imag': np.ascontiguousarray(spec.imag)})

        # numpy ISTFT
        spec_pred = spec * (mask_r + 1j * mask_i)
        out_len = len(buf)
        harmonic = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)
        for t in range(n_frames):
            idx = t * hop_length
            frame = np.fft.irfft(spec_pred[0, 0, :, t], n=n_fft).real * window
            harmonic[idx:idx + n_fft] += frame
            norm[idx:idx + n_fft] += window ** 2
        harmonic = harmonic / np.maximum(norm, 1e-10)

        harmonic = harmonic[pad_len + Tl_pad:pad_len + Tl_pad + n_samples]
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
        # pt2 模型: spec → mask，用 numpy 做 STFT/ISTFT
        n_fft, hop_length = 2048, 512
        seg_length = 32 * hop_length
        window = np.hanning(n_fft).astype(np.float32)

        n_samples = len(wav)
        T1 = n_samples + hop_length
        T_pad = seg_length * ((T1 - 1) // seg_length + 1) - T1
        nl_pad = T_pad // 2 // hop_length
        Tl_pad = nl_pad * hop_length
        padded = np.pad(wav, (Tl_pad, T_pad - Tl_pad), mode='constant')

        # ── numpy STFT（匹配 torch.stft: reflect padding + rfft）──
        pad_len = n_fft // 2
        buf = np.pad(padded, (pad_len, pad_len), mode='reflect')
        n_frames = (len(buf) - n_fft) // hop_length + 1
        spec = np.zeros((1, 1, n_fft // 2 + 1, n_frames), dtype=np.complex64)
        for t in range(n_frames):
            idx = t * hop_length
            spec[0, 0, :, t] = np.fft.rfft(buf[idx:idx + n_fft] * window)

        # ── ONNX mask 推理 ──
        mask_r, mask_i = session.run(
            ['mask_real', 'mask_imag'],
            {'spec_real': np.ascontiguousarray(spec.real),
             'spec_imag': np.ascontiguousarray(spec.imag)})

        # ── numpy ISTFT（重叠相加法）──
        spec_pred = spec * (mask_r + 1j * mask_i)
        out_len = len(buf)
        harmonic = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)
        for t in range(n_frames):
            idx = t * hop_length
            frame = np.fft.irfft(spec_pred[0, 0, :, t], n=n_fft).real * window
            harmonic[idx:idx + n_fft] += frame
            norm[idx:idx + n_fft] += window ** 2
        harmonic = harmonic / np.maximum(norm, 1e-10)
        # 裁剪 reflect padding 和 seg padding
        harmonic = harmonic[pad_len + Tl_pad:pad_len + Tl_pad + n_samples]
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


