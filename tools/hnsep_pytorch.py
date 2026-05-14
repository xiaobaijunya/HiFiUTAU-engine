"""
HN-SEP PyTorch 推理模块

继承 BaseHnsep，差异只在 PyTorch 模型加载和推理。
公共逻辑（breath/tension 后处理）在 base_hnsep.py 中。
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

from tools.base_hnsep import BaseHnsep
from hnsep.nets import CascadedNet


class PytorchHnsep(BaseHnsep):
    """HN-SEP 谐波/噪声分离器 — PyTorch 版。"""

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

        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        print(f"[PytorchHnsep] 加载 checkpoint: {model_path}")

        with open(config_path) as f:
            args = yaml.safe_load(f)

        self.model = CascadedNet(
            args['n_fft'], args['hop_length'], args['n_out'],
            args['n_out_lstm'], True, is_mono=args.get('is_mono', True),
            fixed_length=True if args.get('fixed_length') is None else args['fixed_length'],
        )
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
        waveform = self._ensure_1d(waveform)
        wav_tensor = torch.from_numpy(waveform) \
            .view(1, 1, -1).to(self.device)

        harmonic_tensor = self.model.predict_fromaudio(wav_tensor)
        harmonic = harmonic_tensor.squeeze().cpu().numpy()

        n_samples = wav_tensor.shape[-1]
        noise = waveform[:n_samples] - harmonic[:n_samples]

        return harmonic, noise
