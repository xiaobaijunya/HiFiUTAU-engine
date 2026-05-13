"""
HN-SEP PyTorch 推理模块

使用原生 PyTorch 版本的 HN-SEP（CascadedNet）对音频进行谐波/噪声分离。
替代 onnxruntime 推理，避免 CUDA DLL 缺失导致的 CPU 回退。

模型输入: waveform (batch_size, n_samples) - 2D float32
模型输出: harmonic (batch_size, n_samples), noise (batch_size, n_samples)
"""

import os
import sys
import yaml
import numpy as np
import torch

# 确保能找到 hnsep 包
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from hnsep.nets import CascadedNet


# 全局模型缓存
_global_hnsep_model = None


def get_global_hnsep_model(model_path: str = None, config_path: str = None,
                            device: str = 'cuda') -> 'PytorchHnsep':
    """获取全局 PyTorch HN-SEP 模型（单例）。"""
    global _global_hnsep_model
    if _global_hnsep_model is None:
        if model_path is None:
            model_path = os.path.join("hnsep_onnx", "model.pt")
        if config_path is None:
            config_path = os.path.join("hnsep_onnx", "config.yaml")
        _global_hnsep_model = PytorchHnsep(model_path, config_path, device)
    return _global_hnsep_model


class PytorchHnsep:
    """
    PyTorch HN-SEP 谐波/噪声分离器。

    封装 CascadedNet，提供与 ONNX 版 hnsep_separate() 兼容的接口。
    """

    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Args:
            model_path:  model.pt 路径 (CascadedNet state_dict)
            config_path: config.yaml 路径
            device:      'cuda' 或 'cpu'
        """
        # ── 设备解析 + 自动回退 ──
        _requested = device.lower()
        if _requested in ('cuda', 'gpu', 'tensorrt', 'trt') and not torch.cuda.is_available():
            print(f"[PytorchHnsep] 警告: CUDA 不可用，自动回退到 CPU")
            _requested = 'cpu'
        self.device = torch.device(_requested if _requested != 'dml' else 'cpu')

        # ── CUDA 加速设置 ──
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        print(f"[PytorchHnsep] 加载 checkpoint: {model_path}")

        # 读取配置
        with open(config_path) as f:
            args = yaml.safe_load(f)

        # 创建模型
        self.model = CascadedNet(
            args['n_fft'],
            args['hop_length'],
            args['n_out'],
            args['n_out_lstm'],
            True,                              # is_complex
            is_mono=args.get('is_mono', True),
            fixed_length=True if args.get('fixed_length') is None else args['fixed_length'],
        )
        # 加载权重
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        self.hop_length = args['hop_length']
        self.n_fft = args['n_fft']

        print(f"[PytorchHnsep] 模型就绪，设备={self.device}")

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def separate(self, waveform: np.ndarray) -> tuple:
        """
        分离音频为谐波和噪声分量。

        Args:
            waveform: np.ndarray, shape (samples,) 或 (1, samples), float32

        Returns:
            harmonic: np.ndarray, shape (samples,) — 谐波分量
            noise:    np.ndarray, shape (samples,) — 噪声/气息分量
        """
        # 确保为 (B, C, T) 格式: (1, 1, n_samples)
        if waveform.ndim == 1:
            wav_tensor = torch.from_numpy(waveform.astype(np.float32)) \
                .view(1, 1, -1).to(self.device)
        elif waveform.ndim == 2:
            wav_tensor = torch.from_numpy(waveform.astype(np.float32)) \
                .unsqueeze(1).to(self.device)
        else:
            wav_tensor = torch.from_numpy(waveform.astype(np.float32)) \
                .view(1, 1, -1).to(self.device)

        # predict_fromaudio 返回谐波分量 (B, C, T)
        harmonic_tensor = self.model.predict_fromaudio(wav_tensor)
        harmonic = harmonic_tensor.squeeze().cpu().numpy()

        # 噪声 = 原始 - 谐波（保持与 ONNX 版一致的长度）
        n_samples = wav_tensor.shape[-1]
        noise = waveform[:n_samples] - harmonic[:n_samples]

        return harmonic, noise

    # ------------------------------------------------------------------
    def __call__(self, waveform: np.ndarray) -> tuple:
        """与 ONNX 版 hnsep_separate(wav, session) 兼容的调用方式。"""
        return self.separate(waveform)
