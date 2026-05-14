"""
隐空间混合拼接器 — 公共基类

所有 splicer（ONNX / PyTorch）共享的配置加载、常数定义和工具方法。
具体差异只体现在模型加载和推理上（part1_encode / part2_synthesize）。
"""

import json
import numpy as np
from scipy.interpolate import interp1d


class BaseSplicer:
    """
    公共基类，封装 splicer 的共有逻辑。

    子类需实现:
      part1_encode(mel_2d: np.ndarray) -> np.ndarray
          输入 (128, T) mel → 输出 (1, 128, T*64) 隐特征
      part2_synthesize(feat: np.ndarray, f0: np.ndarray) -> np.ndarray
          输入 (1, 128, T_feat) 隐特征 + (T,) f0 → 输出 (samples,) 波形
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: config.json 路径
        """
        with open(config_path) as f:
            config = json.load(f)

        self.model_hop = config['hop_size']              # 512
        self.sample_rate = config['sampling_rate']       # 44100
        self.ms_per_frame_model = self.model_hop / self.sample_rate * 1000

        # part1 上采样倍数: 前两层 upsample_rates 乘积 (8*8=64)
        self.feat_upsample = int(np.prod(config['upsample_rates'][:2]))
        self.num_mels = config['num_mels']  # 128

        # 编码器补帧：在 part1 编码前给每个音素的 mel 补帧，让卷积层有上下文
        self.encoder_pad_frames = 8  # mel 帧 (hop=512), ≈93ms
        # 前后补帧数：给 HiFi-GAN 卷积层提供边界上下文
        self.front_pad_frames = 6  # ≈70ms
        self.tail_pad_frames = 4   # ≈46ms

    # ─── 子类需实现的抽象方法 ──────────────────────────────

    def part1_encode(self, mel_2d: np.ndarray) -> np.ndarray:
        """mel → 隐特征 (子类实现)"""
        raise NotImplementedError

    def part2_synthesize(self, feat: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """隐特征 + f0 → 波形 (子类实现)"""
        raise NotImplementedError

    # ─── 共用工具方法 ──────────────────────────────────────

    def _resample_mel(self, mel: np.ndarray, target_frames: int) -> np.ndarray:
        """将 mel 从当前帧数用 cubic 插值重采样到 target_frames。"""
        if mel.shape[1] == target_frames:
            return mel.copy()
        old = np.arange(mel.shape[1])
        new = np.linspace(0, mel.shape[1] - 1, target_frames)
        return interp1d(old, mel, axis=1, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')(new)

    def _encode_one(self, mel: np.ndarray, ratio: float) -> np.ndarray:
        """单个音素的 mel → 编码 → 隐特征。

        Args:
            mel:   (n_mels, frames) 原始 mel (hop=256)
            ratio: hop_length / model_hop

        Returns:
            feat:  (1, 128, T_feat) 隐特征
        """
        if mel.shape[1] == 0:
            return None
        target = max(1, int(mel.shape[1] * ratio))
        mel_512 = self._resample_mel(mel, target)
        enc_pad = self.encoder_pad_frames
        if enc_pad > 0:
            pad_mel = np.repeat(mel_512[:, :1], enc_pad, axis=1)
            mel_512 = np.concatenate([pad_mel, mel_512], axis=1)
        feat = self.part1_encode(mel_512)
        if enc_pad > 0:
            trim_feat = enc_pad * self.feat_upsample
            feat = feat[:, :, trim_feat:]
        return feat

    # ─── 拼接 + 合成流程 ──────────────────────────────────

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
        ratio = hop_length / hop_model

        # ---- 第 1 步: 每个音素 encode ----
        feats = []
        feat_lens_hop512 = []

        for info in phoneme_list:
            feat = self._encode_one(info['mel'], ratio)
            feats.append(feat)
            if feat is not None:
                feat_lens_hop512.append(feat.shape[2] // self.feat_upsample)
            else:
                feat_lens_hop512.append(0)

        # ---- 第 2 步: 在 feat 空间交叉淡化拼接 ----
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

            # 特征空间交叉淡化
            fade_in  = np.linspace(0, 1, overlap_feat).reshape(1, 1, -1)
            fade_out = np.linspace(1, 0, overlap_feat).reshape(1, 1, -1)

            tail  = last_feat[:, :, -overlap_feat:]
            body  = last_feat[:, :, :-overlap_feat]
            head  = feat[:, :, :overlap_feat]
            body2 = feat[:, :, overlap_feat:]

            segments[-1] = body
            segments.append(tail * fade_out + head * fade_in)
            segments.append(body2)

        if segments:
            combined_feat = np.concatenate(segments, axis=2)
        else:
            combined_feat = np.empty((1, 128, 0), dtype=np.float32)

        total_mel_frames = combined_feat.shape[2] // self.feat_upsample
        return combined_feat, total_mel_frames

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
                f0[n:] = f0_np[n-1]
        else:
            f0[:] = default_f0

        # ── 首尾补充帧：给卷积层提供上下文 ──
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

        return self.part2_synthesize(combined_feat, f0)

    # ------------------------------------------------------------------
    def splice_and_synthesize(self, phoneme_list, ms_per_frame_hop, hop_length, f0_np):
        """便捷方法: process + synthesize 一步完成。"""
        feat, _ = self.process(phoneme_list, ms_per_frame_hop, hop_length)
        return self.synthesize(feat, f0_np)

    # ------------------------------------------------------------------
    def splice_and_synthesize_mixed(self, phoneme_list, ms_per_frame_hop, hop_length, f0_np):
        """
        混合拼接模式：每个音素按 splc 标志独立选择拼接方式。

        splc=1 的音素与前一个音素在 mel 域做能量归一化交叉淡化，
        splc=0 的音素保持原 feat 域交叉淡化。
        """
        hop_model = self.model_hop
        ratio = hop_length / hop_model

        n = len(phoneme_list)

        # ── 第 1 步: 扫描 splc 标志，构建分组 ──
        segments_idx = []
        cur = [0]
        for i in range(1, n):
            splc = phoneme_list[i].get('Note_flags', {}).get('splc', 0)
            if splc == 1:
                cur.append(i)
            else:
                segments_idx.append(cur)
                cur = [i]
        segments_idx.append(cur)

        # ── 第 2 步: 处理每组得到 feat ──
        seg_feats = []
        seg_first_idx = []

        for seg in segments_idx:
            seg_first_idx.append(seg[0])

            if len(seg) == 1:
                # 单个音素：独立 encode
                feat = self._encode_one(phoneme_list[seg[0]]['mel'], ratio)
                seg_feats.append(feat)
            else:
                # 多个 splc=1 音素：mel 域能量拼接 → 一次 encode
                seg_mels = []
                for j, idx in enumerate(seg):
                    mel = phoneme_list[idx]['mel']
                    if mel.shape[1] == 0:
                        continue
                    if j == 0:
                        seg_mels.append(mel)
                        continue

                    info = phoneme_list[idx]
                    p0_x = info['envelope']['p0']['x']
                    p1_x = info['envelope']['p1']['x']
                    if p1_x < 0:
                        overlap_ms = abs(p0_x) - abs(p1_x)
                    else:
                        overlap_ms = abs(p1_x) + abs(p0_x)
                    ov = int(overlap_ms / ms_per_frame_hop)
                    ov = min(ov, seg_mels[-1].shape[1], mel.shape[1])

                    if ov <= 0:
                        seg_mels.append(mel)
                        continue

                    tail = seg_mels[-1][:, -ov:]
                    head = mel[:, :ov]
                    body = seg_mels[-1][:, :-ov]
                    body2 = mel[:, ov:]

                    tail_lin = np.exp(tail)
                    head_lin = np.exp(head)
                    # 逐帧能量匹配
                    tail_energy = np.mean(tail_lin ** 2, axis=0)
                    head_energy = np.mean(head_lin ** 2, axis=0)
                    head_energy = np.maximum(head_energy, 1e-12)
                    tail_energy = np.maximum(tail_energy, 1e-12)
                    gain = np.sqrt(tail_energy / head_energy)
                    head_lin = head_lin * gain.reshape(1, -1)

                    # 恒定功率交叉淡化 (sqrt fade)
                    fade_in  = np.sqrt(np.linspace(0, 1, ov)).reshape(1, -1)
                    fade_out = np.sqrt(np.linspace(1, 0, ov)).reshape(1, -1)
                    cross_lin = tail_lin * fade_out + head_lin * fade_in
                    cross = np.log(np.maximum(cross_lin, 1e-12))

                    seg_mels[-1] = body
                    seg_mels.append(cross)
                    seg_mels.append(body2)

                if not seg_mels:
                    seg_feats.append(None)
                    continue

                mel_concat = np.concatenate(seg_mels, axis=1)
                feat = self._encode_one(mel_concat, ratio)
                seg_feats.append(feat)

        # ── 第 3 步: 组间 feat 域交叉淡化 ──
        feat_segments = []
        for si, feat in enumerate(seg_feats):
            if feat is None:
                continue
            if si == 0 or not feat_segments:
                feat_segments.append(feat)
                continue

            info = phoneme_list[seg_first_idx[si]]
            p0_x = info['envelope']['p0']['x']
            p1_x = info['envelope']['p1']['x']
            if p1_x < 0:
                overlap_ms = abs(p0_x) - abs(p1_x)
            else:
                overlap_ms = abs(p1_x) + abs(p0_x)
            ov_frames_512 = round(overlap_ms / self.ms_per_frame_model)
            ov_feat = ov_frames_512 * self.feat_upsample

            last_feat = feat_segments[-1]
            ov_feat = min(ov_feat, last_feat.shape[2], feat.shape[2])

            if ov_feat <= 0:
                feat_segments.append(feat)
                continue

            f_in  = np.linspace(0, 1, ov_feat).reshape(1, 1, -1)
            f_out = np.linspace(1, 0, ov_feat).reshape(1, 1, -1)
            tail  = last_feat[:, :, -ov_feat:]
            body  = last_feat[:, :, :-ov_feat]
            head  = feat[:, :, :ov_feat]
            body2 = feat[:, :, ov_feat:]
            feat_segments[-1] = body
            feat_segments.append(tail * f_out + head * f_in)
            feat_segments.append(body2)

        if not feat_segments:
            return np.zeros(0, dtype=np.float32)

        combined_feat = np.concatenate(feat_segments, axis=2)
        return self.synthesize(combined_feat, f0_np)
