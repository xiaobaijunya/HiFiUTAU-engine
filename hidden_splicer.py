"""
隐空间混合拼接器 (ONNX + GPU/CUDA 版)

使用导出的 part1.onnx / part2.onnx 在 HiFiGAN 隐层特征空间
进行音素拼接，替代 mel 域交叉淡入淡出 + 单次 ONNX 合成。

原理:
  每个音素的 mel 分别经过 part1.onnx 得到中间隐特征，
  在特征空间做交叉淡化拼接，再通过 part2.onnx 合成波形。
"""

import os
import json
import numpy as np
import onnxruntime
from scipy.interpolate import interp1d


class HiddenSplicer:
    """
    隐空间混合拼接器 (ONNX)

    流程:
      1. 每个音素 mel (hop=256) → 裁剪开头 → 重采样到 hop=512
      2. → part1.onnx → 隐特征 feat
      3. 在 feat 空间做交叉淡入淡出拼接
      4. part2.onnx(feat_spliced, f0) → 波形
    """

    def __init__(self, part1_onnx_path, part2_onnx_path, config_path, device='cuda'):
        """
        Args:
            part1_onnx_path: part1.onnx 路径 (mel → 隐特征)
            part2_onnx_path: part2.onnx 路径 (隐特征 + f0 → 波形)
            config_path:     config.json 路径
            device:          'cuda' 或 'cpu'
        """
        # 加载模型配置
        with open(config_path) as f:
            config = json.load(f)
        self.model_hop = config['hop_size']          # 512
        self.sample_rate = config['sampling_rate']   # 44100
        self.ms_per_frame_model = self.model_hop / self.sample_rate * 1000

        # part1 上采样倍数: 前两层 upsample_rates 乘积 (8*8=64)
        self.feat_upsample = int(np.prod(config['upsample_rates'][:2]))
        self.num_mels = config['num_mels']  # 128

        # ONNX Runtime 会话 (优先 CUDA)
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device.startswith('cuda') else ['CPUExecutionProvider'])
        self.part1_session = onnxruntime.InferenceSession(part1_onnx_path, providers=providers)
        self.part2_session = onnxruntime.InferenceSession(part2_onnx_path, providers=providers)

        print(f'[HiddenSplicer] ONNX 模型已加载, device={device}')

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
    def process(self, phoneme_list, ms_per_frame_hop, hop_length):
        """
        对音素列表做隐空间拼接，返回拼接后的特征数组。

        Args:
            phoneme_list: list[dict]
                每个 dict 含 'mel' (n_mels, frames, hop=256), 'envelope', 等。
            ms_per_frame_hop: float, hop=256 时的 ms/帧

        Returns:
            combined_feat:     np.ndarray (1, 128, T_feat)
            total_mel_frames:  int, 拼接后对应 hop=512 的 mel 帧数
        """
        hop_model = self.model_hop  # 512

        # ---- 第 1 步: 每个音素 trim → resample → part1.onnx ----
        feats = []            # list of (1, 128, T*64) or None
        feat_lens_hop512 = [] # 每个音素 mel 在 hop=512 下的帧数

        for info in phoneme_list:
            mel = info['mel']
            if mel.shape[1] == 0:
                feats.append(None)
                feat_lens_hop512.append(0)
                continue

            # --- 重采样 hop=256 → hop=512 ---
            ratio = hop_length / hop_model
            target = max(1, round(mel.shape[1] * ratio))
            mel_hop512 = self._resample_mel(mel, target)
            feat_lens_hop512.append(mel_hop512.shape[1])

            # --- part1.onnx → 隐特征 ---
            mel_input = np.expand_dims(mel_hop512.astype(np.float32), axis=0)  # (1, 128, T)
            feat = self.part1_session.run(['feat'], {'mel': mel_input})[0]     # (1, 128, T*64)
            feats.append(feat)

        # ---- 第 2 步: 在 feat 空间交叉淡化拼接 (numpy) ----
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

            # 特征空间交叉淡化 (numpy)
            fade_in  = np.linspace(0, 1, overlap_feat).reshape(1, 1, -1)
            fade_out = np.linspace(1, 0, overlap_feat).reshape(1, 1, -1)

            tail  = last_feat[:, :, -overlap_feat:]
            body  = last_feat[:, :, :-overlap_feat]
            head  = feat[:, :, :overlap_feat]
            body2 = feat[:, :, overlap_feat:]

            segments[-1] = body
            segments.append(tail * fade_out + head * fade_in)
            segments.append(body2)

        # 拼合
        if segments:
            combined_feat = np.concatenate(segments, axis=2)
        else:
            combined_feat = np.empty((1, 128, 0), dtype=np.float32)

        total_mel_frames = combined_feat.shape[2] // self.feat_upsample
        return combined_feat, total_mel_frames

    # ------------------------------------------------------------------
    def synthesize(self, combined_feat, f0_np, default_f0=440.0):
        """
        从拼接好的隐特征合成波形。

        Args:
            combined_feat: np.ndarray (1, 128, T_feat)
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

        # ONNX 推理 part2: 隐特征 + f0 → 波形
        f0_input = f0.reshape(1, -1).astype(np.float32)  # (1, T_mel)
        feat_input = combined_feat.astype(np.float32)     # (1, 128, T_feat)

        wav = self.part2_session.run(
            ['waveform'], {'feat': feat_input, 'f0': f0_input}
        )[0]  # (1, 1, samples)

        return wav.squeeze()

    # ------------------------------------------------------------------
    def splice_and_synthesize(self, phoneme_list, ms_per_frame_hop, hop_length, f0_np):
        """
        便捷方法: process + synthesize 一步完成。

        Args:
            phoneme_list:        list[dict], 同 process()
            ms_per_frame_hop256: float, hop=256 的 ms/帧
            f0_np:               np.ndarray, 重采样到 hop=512 的 F0

        Returns:
            waveform: np.ndarray (samples,)
        """
        feat, _ = self.process(phoneme_list, ms_per_frame_hop, hop_length)
        return self.synthesize(feat, f0_np)
