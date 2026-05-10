"""
隐空间混合拼接器

使用 SplitGenerator 在 HiFiGAN 隐层特征空间进行音素拼接，
替代 mel 域交叉淡入淡出 + ONNX 合成的方式。

原理:
  每个音素的 mel 分别经过 SplitGenerator.forward_part1() 
  得到中间隐特征，在特征空间做交叉淡化拼接，
  再通过 forward_part2() 合成波形。
"""

import os
import sys
import json
import pathlib
import numpy as np
import torch
from scipy.interpolate import interp1d

from tools.nsf_hifigan import SplitGenerator, AttrDict


class HiddenSplicer:
    """
    隐空间混合拼接器

    流程:
      1. 每个音素 mel (hop=256) → 裁剪开头 → 重采样到 hop=512
      2. → SplitGenerator.forward_part1() → 隐特征 feat
      3. 在 feat 空间做交叉淡入淡出拼接
      4. forward_part2(feat_spliced, f0) → 波形
    """

    def __init__(self, ckpt_path, device='cpu'):
        self.device = torch.device(device)

        # 加载模型配置
        config_file = os.path.join(os.path.dirname(ckpt_path), 'config.json')
        with open(config_file) as f:
            h = AttrDict(json.load(f))
        self.h = h
        self.model_hop = h.hop_size          # 512
        self.sample_rate = h.sampling_rate   # 44100
        self.ms_per_frame_model = self.model_hop / self.sample_rate * 1000

        # part1 上采样倍数: 前两层 upsample_rates 乘积 (8*8=64)
        self.feat_upsample = int(np.prod(h.upsample_rates[:2]))

        # 构建模型
        self.generator = SplitGenerator(h)
        cp_dict = torch.load(ckpt_path, map_location='cpu')
        self.generator.load_state_dict(cp_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.generator.to(self.device)
        del cp_dict
        print(f'[HiddenSplicer] 模型已加载到 {device}')

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
    def process(self, phoneme_list, ms_per_frame_hop,hop_length):
        """
        对音素列表做隐空间拼接，返回拼接后的特征张量。

        Args:
            phoneme_list: list[dict]
                每个 dict 含 'mel' (n_mels, frames, hop=256), 'envelope', 等。
            ms_per_frame_hop: float, hop=256 时的 ms/帧

        Returns:
            combined_feat:        torch.Tensor (1, 128, T_feat)
            total_mel_frames:     int, 拼接后对应 hop=512 的 mel 帧数
        """
        hop_model = self.model_hop  # 512

        # ---- 第 1 步: 每个音素 trim → resample → part1 ----
        feats = []            # list of (1, 128, T*64) or None
        feat_lens_hop512 = [] # 每个音素 mel 在 hop=512 下的帧数

        for info in phoneme_list:
            mel = info['mel']
            if mel.shape[1] == 0:
                feats.append(None)
                feat_lens_hop512.append(0)
                continue

            # mel 开头已在 cut_audio() 中裁剪完毕，此处无需重复裁剪

            if mel.shape[1] == 0:
                feats.append(None)
                feat_lens_hop512.append(0)
                continue

            # --- 重采样 hop=256 → hop=512 ---
            ratio = hop_length / hop_model
            target = max(1, round(mel.shape[1] * ratio))
            mel_hop512 = self._resample_mel(mel, target)
            feat_lens_hop512.append(mel_hop512.shape[1])

            # --- part1 → 隐特征 ---
            t = torch.from_numpy(mel_hop512).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.generator.forward_part1(t)  # (1, 128, T*64)
            feats.append(feat.cpu())

        # ---- 第 2 步: 在 feat 空间交叉淡化拼接 ----
        segments = []  # list of (1, 128, T_feat)

        for i in range(len(feats)):
            feat = feats[i]
            if feat is None:
                continue

            if i == 0 or not segments:
                segments.append(feat)
                continue

            # 计算 overlap 毫秒 → feat 帧数
            info = phoneme_list[i]
            p0_x = info['envelope']['p0']['x']
            p1_x = info['envelope']['p1']['x']

            if p1_x < 0:
                overlap_ms = abs(p0_x) - abs(p1_x)
            else:
                overlap_ms = abs(p1_x) + abs(p0_x)

            overlap_frames_h512 = round(overlap_ms / self.ms_per_frame_model)
            overlap_feat = overlap_frames_h512 * self.feat_upsample

            last_feat = segments[-1]
            overlap_feat = min(overlap_feat, last_feat.shape[2], feat.shape[2])

            if overlap_feat <= 0:
                segments.append(feat)
                continue

            # 特征空间交叉淡化
            fade_in  = torch.linspace(0, 1, overlap_feat).view(1, 1, -1)
            fade_out = torch.linspace(1, 0, overlap_feat).view(1, 1, -1)

            tail  = last_feat[:, :, -overlap_feat:]
            body  = last_feat[:, :, :-overlap_feat]
            head  = feat[:, :, :overlap_feat]
            body2 = feat[:, :, overlap_feat:]

            segments[-1] = body
            segments.append(tail * fade_out + head * fade_in)
            segments.append(body2)

        # 拼合
        if segments:
            combined_feat = torch.cat(segments, dim=2)
        else:
            combined_feat = torch.empty((1, 128, 0))

        total_mel_frames = combined_feat.shape[2] // self.feat_upsample
        return combined_feat, total_mel_frames

    # ------------------------------------------------------------------
    def synthesize(self, combined_feat, f0_np, default_f0=440.0):
        """
        从拼接好的隐特征合成波形。

        Args:
            combined_feat: torch.Tensor (1, 128, T_feat)
            f0_np:         np.ndarray, 重采样到 hop=512 的 F0 序列
            default_f0:    float, 无 F0 时的默认值

        Returns:
            waveform: np.ndarray (samples,)
        """
        mel_frames_needed = combined_feat.shape[2] // self.feat_upsample

        # 补齐 / 截断 F0 到所需长度（复制最后一帧）
        f0 = np.zeros(mel_frames_needed, dtype=np.float32)
        n = min(len(f0_np), mel_frames_needed)
        if n > 0:
            f0[:n] = f0_np[:n]
            if n < mel_frames_needed:
                f0[n:] = f0_np[n-1]  # 复制最后一帧
        else:
            f0[:] = default_f0

        f0_t = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)  # (1, T)
        feat_t = combined_feat.to(self.device)

        with torch.no_grad():
            wav_t, _, _ = self.generator.forward_part2(
                feat_t, f0_t, nsf_gain=1.0, x_gain=1.0, out_har=True
            )

        return wav_t.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    def splice_and_synthesize(self, phoneme_list, ms_per_frame_hop,hop_length, f0_np):
        """
        便捷方法: process + synthesize 一步完成。

        Args:
            phoneme_list:        list[dict], 同 process()
            ms_per_frame_hop256: float, hop=256 的 ms/帧
            f0_np:               np.ndarray, 重采样到 hop=512 的 F0

        Returns:
            waveform: np.ndarray (samples,)
        """
        feat, _ = self.process(phoneme_list, ms_per_frame_hop,hop_length)
        return self.synthesize(feat, f0_np)
