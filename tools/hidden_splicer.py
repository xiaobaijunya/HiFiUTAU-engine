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

    def __init__(self, part1_onnx_path, part2_onnx_path, config_path, device='dml'):
        """
        Args:
            part1_onnx_path: part1.onnx 路径 (mel → 隐特征)
            part2_onnx_path: part2.onnx 路径 (隐特征 + f0 → 波形)
            config_path:     config.json 路径
            device:          'dml' (DirectML, 默认), 'cuda', 或 'cpu'
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

        # 编码器补帧：在 part1 编码前给每个音素的 mel 补帧，让卷积层有上下文
        self.encoder_pad_frames = 8  # mel 帧 (hop=512), ≈93ms
        # 前后补帧数：给 HiFi-GAN 卷积层提供边界上下文
        self.front_pad_frames = 6  # ≈70ms
        self.tail_pad_frames = 4   # ≈46ms

        # ONNX Runtime 会话 — 根据 device 选择 provider
        providers = self._resolve_providers(device)
        self.part1_session = onnxruntime.InferenceSession(part1_onnx_path, providers=providers)
        self.part2_session = onnxruntime.InferenceSession(part2_onnx_path, providers=providers)

        print(f'[HiddenSplicer] ONNX 模型已加载, providers={providers}')

    @staticmethod
    def _resolve_providers(device: str) -> list:
        """根据 device 名称解析 ONNX Runtime provider 列表（带可用性检测+回退）。"""
        device = device.lower()
        if device == 'cpu':
            return ['CPUExecutionProvider']

        available = onnxruntime.get_available_providers()

        if device in ('dml', 'directml'):
            if 'DmlExecutionProvider' in available:
                return ['DmlExecutionProvider', 'CPUExecutionProvider']
            print('[警告] DmlExecutionProvider 不可用，回退到 CPU')
            return ['CPUExecutionProvider']

        if device in ('cuda', 'gpu'):
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print('[警告] CUDAExecutionProvider 不可用，回退到 CPU')
            return ['CPUExecutionProvider']

        print(f'[警告] 未知 device="{device}"，回退到 CPU')
        return ['CPUExecutionProvider']

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

            # --- 重采样 hop=44 → hop=512 ---
            ratio = hop_length / hop_model
            target = max(1, int(mel.shape[1] * ratio))
            mel_hop512 = self._resample_mel(mel, target)

            # 在 part1 编码前给 mel 开头补帧，提供卷积层上下文
            enc_pad = self.encoder_pad_frames
            if enc_pad > 0:
                pad_mel = np.repeat(mel_hop512[:, :1], enc_pad, axis=1)
                mel_hop512 = np.concatenate([pad_mel, mel_hop512], axis=1)

            feat_lens_hop512.append(mel_hop512.shape[1] - enc_pad)

            # --- part1.onnx → 隐特征 ---
            mel_input = np.expand_dims(mel_hop512.astype(np.float32), axis=0)  # (1, 128, T+pad)
            feat = self.part1_session.run(['feat'], {'mel': mel_input})[0]     # (1, 128, (T+pad)*64)

            # 裁掉编码器补帧对应的特征
            if enc_pad > 0:
                trim_feat = enc_pad * self.feat_upsample
                feat = feat[:, :, trim_feat:]

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

        原理：HiFi-GAN 卷积层需要尾部"未来上下文"才能稳定解码。
        末尾补 self.tail_pad_frames 帧虚假内容让卷积层有上下文，
        裁剪延后到 SynthesisEngine 中 HN-SEP 之后统一处理。

        Args:
            combined_feat:  np.ndarray (1, 128, T_feat)
            f0_np:          np.ndarray, 重采样到 hop=512 的 F0 序列
            default_f0:     float, 无 F0 时的默认值

        Returns:
            waveform: np.ndarray (samples,) — 含尾部补帧，由调用方裁剪
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

        # ── 首尾补充帧：给卷积层提供上下文 ──
        # 前面补帧让 HiFi-GAN 有"预热"上下文，避免开头采样被零填充拉低
        # 尾部补帧让卷积层有"未来上下文"
        # 统一在 SynthesisEngine 中裁剪
        if self.front_pad_frames > 0:
            pad_feat = self.front_pad_frames * self.feat_upsample
            first_feat = combined_feat[:, :, :1]
            pad_front = np.repeat(first_feat, pad_feat, axis=2)
            combined_feat = np.concatenate([pad_front, combined_feat], axis=2)
            pad_f0_front = np.full(self.front_pad_frames, f0[0], dtype=np.float32)
            f0 = np.concatenate([pad_f0_front, f0])
        if self.tail_pad_frames > 0:
            pad_feat = self.tail_pad_frames * self.feat_upsample
            last_feat = combined_feat[:, :, -1:]
            pad_tail = np.repeat(last_feat, pad_feat, axis=2)
            combined_feat = np.concatenate([combined_feat, pad_tail], axis=2)
            pad_f0_tail = np.full(self.tail_pad_frames, f0[-1], dtype=np.float32)
            f0 = np.concatenate([f0, pad_f0_tail])

        # ONNX 推理 part2: 隐特征 + f0 → 波形
        f0_input = f0.reshape(1, -1).astype(np.float32)
        feat_input = combined_feat.astype(np.float32)

        wav = self.part2_session.run(
            ['waveform'], {'feat': feat_input, 'f0': f0_input}
        )[0]

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
