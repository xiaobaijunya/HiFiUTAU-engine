"""
隐空间混合拼接器 (PyTorch 优化版)

使用原始的 PyTorch SplitGenerator checkpoint，替代 ONNX Runtime 推理。
与 tools.hidden_splicer.py 接口完全兼容，可无缝替换。

优化措施:
  1. 全流程 GPU 驻留 — 消除 part1→part2 间的 CPU↔GPU 拷贝
  2. 交叉淡化在 GPU 上完成 (torch)
  3. torch.inference_mode() 替代 no_grad()，减少 autograd 开销
  4. 支持 torch.compile — 将 model 编译为优化内核 (2~4x 加速)
  5. 支持 FP16 推理 — 显存减半 + 吞吐提升
"""

import os
import json
import sys
import numpy as np
import torch
from scipy.interpolate import interp1d

# 确保能找到 tools 包
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import librosa

from tools.nsf_hifigan import SplitGenerator, AttrDict
from tools.utils import init_weights, get_padding


class PytorchHiddenSplicer:
    """
    隐空间混合拼接器 (PyTorch 优化版)

    与 ONNX 版 HiddenSplicer 接口完全一致，但使用原生 PyTorch 推理。

    流程:
      1. 每个音素 mel (hop=256) → 裁剪开头 → 重采样到 hop=512
      2. → SplitGenerator.forward_part1() → 隐特征 feat (全程 GPU)
      3. 在 feat 空间做交叉淡入淡出拼接 (GPU)
      4. SplitGenerator.forward_part2(feat_spliced, f0) → 波形 (GPU)
    """

    def __init__(self, checkpoint_path, config_path, device='cuda',
                 compile_model=False, fp16=False, griffin_lim_mode=False):
        """
        Args:
            checkpoint_path: model.ckpt 路径 (PyTorch SplitGenerator 权重)
            config_path:     config.json 路径
            device:          'cuda', 'cpu', 等 PyTorch 设备名
            compile_model:   是否使用 torch.compile (需 PyTorch ≥2.0)
            fp16:            是否使用 FP16 推理 (仅 GPU 有效)
            griffin_lim_mode: 为 True 时跳过 HiFiGAN，使用 Griffin-Lim 重建波形（测试用）
        """
        self.griffin_lim_mode = griffin_lim_mode
        # 加载模型配置
        with open(config_path) as f:
            config = json.load(f)
        self.h = AttrDict(config)
        self.model_hop = self.h.hop_size           # 512
        self.sample_rate = self.h.sampling_rate     # 44100
        self.ms_per_frame_model = self.model_hop / self.sample_rate * 1000

        # part1 上采样倍数: 前两层 upsample_rates 乘积 (8*8=64)
        self.feat_upsample = int(np.prod(self.h.upsample_rates[:2]))
        self.num_mels = self.h.num_mels  # 128

        # 编码器补帧：在 part1 编码前给每个音素的 mel 补帧，让卷积层有上下文
        self.encoder_pad_frames = 8  # mel 帧 (hop=512), ≈93ms
        # 前后补帧数：给 HiFi-GAN 卷积层提供边界上下文
        self.front_pad_frames = 6  # ≈70ms
        self.tail_pad_frames = 4   # ≈46ms

        self.fp16 = fp16 and device.lower() != 'cpu'

        # ── CUDA 加速设置 ──
        # ── 设备解析 + 自动回退 ──
        _requested = device.lower()
        if _requested in ('cuda', 'gpu', 'tensorrt', 'trt') and not torch.cuda.is_available():
            print(f"[警告] CUDA 不可用，自动回退到 CPU")
            _requested = 'cpu'
        self.device = torch.device(_requested if _requested != 'dml' else 'cpu')

        if torch.cuda.is_available() and self.device.type == 'cuda':
            # TF32: Ampere+ GPU 上 FP32 矩阵乘 ≈8x 加速，精度无损
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        print(f"[PytorchHiddenSplicer] 加载 checkpoint: {checkpoint_path}")
        print(f"[PytorchHiddenSplicer] 推理设备: {self.device}")

        cp_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model = SplitGenerator(self.h)
        self.model.load_state_dict(cp_dict['generator'])
        self.model.eval()
        self.model.remove_weight_norm()
        self.model.to(self.device)
        if self.fp16:
            self.model = self.model.half()
        del cp_dict

        # ── torch.compile（可选） ──
        self._compiled = False
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                self._compiled = True
                print("[PytorchHiddenSplicer] torch.compile 已启用 (mode=max-autotune)")
            except Exception as e:
                print(f"[PytorchHiddenSplicer] torch.compile 失败，回退到 eager mode: {e}")

        print(f"[PytorchHiddenSplicer] 模型就绪，设备={self.device}"
              f"{' fp16' if self.fp16 else ''}"
              f"{' compiled' if self._compiled else ''}")

    # ------------------------------------------------------------------
    def _resample_mel(self, mel: np.ndarray, target_frames: int) -> np.ndarray:
        """将 mel 从当前帧数用 cubic 插值重采样到 target_frames。"""
        if mel.shape[1] == target_frames:
            return mel.copy()
        old = np.arange(mel.shape[1])
        new = np.linspace(0, mel.shape[1] - 1, target_frames)
        return interp1d(old, mel, axis=1, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')(new)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process(self, phoneme_list, ms_per_frame_hop, hop_length):
        """
        对音素列表做隐空间拼接，返回拼接后的特征张量（仍在 GPU 上）。

        Args:
            phoneme_list: list[dict]
                每个 dict 含 'mel' (n_mels, frames, hop=256), 'envelope', 等。
            ms_per_frame_hop: float, hop=256 时的 ms/帧

        Returns:
            combined_feat:  torch.Tensor (1, 128, T_feat) — 驻留在 self.device
            total_mel_frames: int
        """
        hop_model = self.model_hop  # 512
        dtype = torch.float16 if self.fp16 else torch.float32

        # ---- 第 1 步: 每个音素 resample → part1 (GPU) ----
        feats = []            # list of torch.Tensor(1,128,T*64) or None
        feat_lens_hop512 = []

        for info in phoneme_list:
            mel = info['mel']
            if mel.shape[1] == 0:
                feats.append(None)
                feat_lens_hop512.append(0)
                continue

            # 重采样 hop=44 → hop=512 (numpy)
            ratio = hop_length / hop_model
            target = max(1, int(mel.shape[1] * ratio))
            mel_hop512 = self._resample_mel(mel, target)

            # 编码器补帧
            enc_pad = self.encoder_pad_frames
            if enc_pad > 0:
                pad_mel = np.repeat(mel_hop512[:, :1], enc_pad, axis=1)
                mel_hop512 = np.concatenate([pad_mel, mel_hop512], axis=1)

            feat_lens_hop512.append(mel_hop512.shape[1] - enc_pad)

            # 送 GPU → forward_part1 （保持 dtype）
            mel_tensor = torch.from_numpy(mel_hop512.astype(np.float32)) \
                .unsqueeze(0).to(device=self.device, dtype=dtype)
            feat_tensor = self.model.forward_part1(mel_tensor)

            # 裁掉编码器补帧对应的特征
            if enc_pad > 0:
                trim_feat = enc_pad * self.feat_upsample
                feat_tensor = feat_tensor[:, :, trim_feat:]

            feats.append(feat_tensor)

        # ---- 第 2 步: 在 feat 空间交叉淡化拼接 (GPU) ----
        segments = []

        for i in range(len(feats)):
            feat = feats[i]
            if feat is None:
                continue

            if i == 0 or not segments:
                segments.append(feat)
                continue

            # 计算 overlap
            info = phoneme_list[i]
            p0_x = info['envelope']['p0']['x']
            p1_x = info['envelope']['p1']['x']
            if p1_x < 0:
                overlap_ms = abs(p0_x) - abs(p1_x)
            else:
                overlap_ms = abs(p1_x) + abs(p0_x)

            overlap_frames_h512 = int(overlap_ms / self.ms_per_frame_model)
            overlap_feat = overlap_frames_h512 * self.feat_upsample

            last_feat = segments[-1]
            overlap_feat = min(overlap_feat, last_feat.shape[2], feat.shape[2])

            if overlap_feat <= 0:
                segments.append(feat)
                continue

            # GPU 交叉淡化
            fade_in  = torch.linspace(0, 1, overlap_feat, device=self.device, dtype=dtype) \
                .view(1, 1, -1)
            fade_out = torch.linspace(1, 0, overlap_feat, device=self.device, dtype=dtype) \
                .view(1, 1, -1)

            tail  = last_feat[:, :, -overlap_feat:]
            body  = last_feat[:, :, :-overlap_feat]
            head  = feat[:, :, :overlap_feat]
            body2 = feat[:, :, overlap_feat:]

            segments[-1] = body
            segments.append(tail * fade_out + head * fade_in)
            segments.append(body2)

        if segments:
            combined_feat = torch.cat(segments, dim=2)
        else:
            combined_feat = torch.empty(1, 128, 0, device=self.device, dtype=dtype)

        total_mel_frames = combined_feat.shape[2] // self.feat_upsample
        return combined_feat, total_mel_frames

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def synthesize(self, combined_feat, f0_np, default_f0=440.0):
        """
        从拼接好的隐特征合成波形。

        Args:
            combined_feat:  torch.Tensor (1, 128, T_feat) — GPU 张量
            f0_np:          np.ndarray, 重采样到 hop=512 的 F0 序列
            default_f0:     float, 无 F0 时的默认值

        Returns:
            waveform: np.ndarray (samples,) — 含尾部补帧，由调用方裁剪
        """
        dtype = combined_feat.dtype
        mel_frames_needed = combined_feat.shape[2] // self.feat_upsample

        # 补齐/截断 F0（在 GPU 上操作）
        f0 = torch.zeros(mel_frames_needed, device=self.device, dtype=dtype)
        n = min(len(f0_np), mel_frames_needed)
        if n > 0:
            f0_t = torch.from_numpy(f0_np.astype(np.float32)).to(self.device)
            f0[:n] = f0_t[:n].to(dtype)
            if n < mel_frames_needed:
                f0[n:] = f0[n-1]  # 复制最后一帧
        else:
            f0[:] = default_f0

        # ── 首尾补充帧（GPU 上操作） ──
        if self.front_pad_frames > 0:
            pad_feat = self.front_pad_frames * self.feat_upsample
            pad_front = combined_feat[:, :, :1].expand(-1, -1, pad_feat)
            combined_feat = torch.cat([pad_front, combined_feat], dim=2)
            pad_f0_front = f0[:1].expand(self.front_pad_frames)
            f0 = torch.cat([pad_f0_front, f0])
        if self.tail_pad_frames > 0:
            pad_feat = self.tail_pad_frames * self.feat_upsample
            pad_tail = combined_feat[:, :, -1:].expand(-1, -1, pad_feat)
            combined_feat = torch.cat([combined_feat, pad_tail], dim=2)
            pad_f0_tail = f0[-1:].expand(self.tail_pad_frames)
            f0 = torch.cat([f0, pad_f0_tail])

        # ── forward_part2: 隐特征 + f0 → 波形 ──
        f0_input = f0.unsqueeze(0)  # (1, T)
        wav_tensor = self.model.forward_part2(
            combined_feat, f0_input,
            nsf_gain=1.0, x_gain=1.0, out_har=False,
        )

        return wav_tensor.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    def splice_and_synthesize(self, phoneme_list, ms_per_frame_hop, hop_length, f0_np):
        """
        便捷方法: process + synthesize 一步完成（全程 GPU）。

        Args:
            phoneme_list:        list[dict], 同 process()
            ms_per_frame_hop256: float, hop=256 的 ms/帧
            f0_np:               np.ndarray, 重采样到 hop=512 的 F0

        Returns:
            waveform: np.ndarray (samples,)
        """
        if self.griffin_lim_mode:
            return self._griffin_lim_synthesize(phoneme_list, hop_length)
        feat, _ = self.process(phoneme_list, ms_per_frame_hop, hop_length)
        return self.synthesize(feat, f0_np)

    def _griffin_lim_synthesize(self, phoneme_list, hop_length):
        """
        临时测试：用 Griffin-Lim 替代 HiFiGAN 从 mel 重建波形。
        忽略 F0，仅用于验证节奏/长度问题是否来自 HiFiGAN。

        Args:
            phoneme_list:  list[dict], 同 process()
            hop_length:    原始 hop（44）

        Returns:
            waveform: np.ndarray (samples,)
        """
        sr = self.sample_rate  # 44100
        n_fft = 2048
        hop = self.model_hop   # 512
        win_length = 2048

        # 收集所有音素的 mel，重采样到 hop=512 并拼接
        mels_512 = []
        for info in phoneme_list:
            mel = info['mel']
            if mel.shape[1] == 0:
                continue
            ratio = hop_length / hop  # 44/512
            target = max(1, int(mel.shape[1] * ratio))
            mel_512 = self._resample_mel(mel, target)
            mels_512.append(mel_512)

        if not mels_512:
            return np.zeros(0, dtype=np.float32)

        mel_concat = np.concatenate(mels_512, axis=1)  # (128, T)

        # log-mel → 线性幅度 mel
        mel_linear = np.exp(mel_concat.astype(np.float64))
        mel_linear = np.clip(mel_linear, 1e-12, None)

        print(f"[Griffin-Lim] mel 帧数: {mel_concat.shape[1]}, 重建中...")
        y = librosa.feature.inverse.mel_to_audio(
            mel_linear, sr=sr, n_fft=n_fft, hop_length=hop,
            win_length=win_length, window='hann',
            center=True, power=1.0, n_iter=32,
            fmin=40, fmax=16000,
        )
        print(f"[Griffin-Lim] 波形长度: {len(y)} 采样 ({len(y)/sr*1000:.1f}ms)")
        return y
