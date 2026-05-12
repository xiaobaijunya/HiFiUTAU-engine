"""
Fragment — 音素片段处理：音频切割、mel 转换、时间拉伸。

cut_audio 使用 ThreadPoolExecutor 并行读取+处理多个音频文件。
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
from synthesis_pipeline.utils import read_audio, dynamic_range_compression, resample_array, interp_to_len


class Fragment:
    """解析 JSON 并处理音频片段。"""

    def __init__(self, json_data: dict):
        self.sample_rate = 44100
        self.hop_length = 44
        self.ms_per_frame = (self.hop_length / self.sample_rate) * 1000

        self.Dynamic_hop = json_data['hop_size']
        self.out_wav = json_data['out_wav']
        self.wav_dur = json_data['wav_dur']
        self.phoneme_list = list(json_data['phoneme_list'].values())

        dp = json_data['Dynamic_parameter']
        self.pit = self._get_param(dp, ('pit', 'pitd'))
        self.tension = self._get_param(dp, ('tension', 'tenc'))
        self.breath = self._get_param(dp, ('breath', 'brec'))
        self.voicing = self._get_param(dp, ('voicing', 'voic'))
        self.growl = self._get_param(dp, ('growl','gwl'))
        self.lowcut = self._get_param(dp, ('lowcut', 'lowc'))
        self.brel = self._get_param(dp, ('brel', 'bret_low'))
        self.breh = self._get_param(dp, ('breh', 'bret_high'))

    # ─── 静态工具 ───
    @staticmethod
    def _get_param(dp: dict, keys: tuple) -> np.ndarray:
        for k in keys:
            val = dp.get(k)
            if val is not None:
                return np.array(val, dtype=np.float32)
        return np.array([], dtype=np.float32)

    # ─── 参数范围计算 ───
    def _get_param_range(self, phoneme_idx: int) -> tuple:
        total_params = len(self.gen) if len(self.gen) > 0 else 1
        cum_frames = 0.0
        for i in range(phoneme_idx):
            mel = self.phoneme_list[i].get('mel')
            if mel is not None:
                cum_frames += mel.shape[1]
            # 减去前一个音素与本音素的重叠帧（即本音素开头被前音素覆盖的部分）
            if i > 0:
                env = self.phoneme_list[i]['envelope']
                p0_x, p1_x = env['p0']['x'], env['p1']['x']
                if p1_x < 0:
                    overlap_ms = abs(p0_x) - abs(p1_x)
                else:
                    overlap_ms = abs(p1_x) + abs(p0_x)
                cum_frames -= overlap_ms / self.ms_per_frame

        # 也减去当前音素与前一个音素的重叠（上一轮循环没处理的最后一段重叠）
        if phoneme_idx > 0:
            env = self.phoneme_list[phoneme_idx]['envelope']
            p0_x, p1_x = env['p0']['x'], env['p1']['x']
            if p1_x < 0:
                overlap_ms = abs(p0_x) - abs(p1_x)
            else:
                overlap_ms = abs(p1_x) + abs(p0_x)
            cum_frames -= overlap_ms / self.ms_per_frame

        cur_mel = self.phoneme_list[phoneme_idx].get('mel')
        cur_frames = cur_mel.shape[1] if cur_mel is not None else 0

        start_frame = int(round(max(0, cum_frames)))
        end_frame = int(round(min(total_params, cum_frames + cur_frames)))
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        return start_frame, end_frame

    def _get_avg_param(self, param_array: np.ndarray, phoneme_idx: int,
                        default: float = 0.0) -> float:
        if len(param_array) == 0:
            return default
        s, e = self._get_param_range(phoneme_idx)
        return float(np.mean(param_array[s:e]))

    # ─── 单音素处理（供并行调用） ───
    def _process_single_phoneme(self, i: int) -> int:
        """处理单个音素：读取音频 → mel 转换 → 时间拉伸。返回音素索引。"""
        info = self.phoneme_list[i]
        oto = info['phoneme_oto']
        wav_path = oto['audio_file_path']

        if not os.path.exists(wav_path):
            print(f'音频文件不存在: {wav_path}')
            return i

        sr = self.sample_rate
        n_fft = 2048
        base_hop = self.hop_length
        win_length = 2048

        # 读取 + 重采样
        audio = read_audio(wav_path, sr)
        total_len_ms = len(audio) / sr * 1000

        offset_ms = oto['Offset']
        consonant_ms = oto['Consonant']
        cutoff_ms = oto['Cutoff']
        Preutter_ms = oto['Preutter']

        # 切段（使用 round 避免 sample 边界系统性偏移）
        start_sample = int(round(offset_ms / 1000 * sr))
        consonant_sample = int(round((offset_ms + consonant_ms) / 1000 * sr))
        if cutoff_ms > 0:
            end_sample = int(round((total_len_ms - cutoff_ms) / 1000 * sr))
        else:
            end_sample = int(round((offset_ms + abs(cutoff_ms)) / 1000 * sr))

        start_sample = max(0, min(start_sample, len(audio)))
        consonant_sample = max(start_sample, min(consonant_sample, len(audio)))
        end_sample = max(consonant_sample, min(end_sample, len(audio)))

        audio_seg = audio[start_sample:end_sample]
        con_samples = consonant_sample - start_sample

        # vol（保存值，实际在 engine.py 中 P→phtp→gen 之后统一应用）
        vol = info.get('Note_flags', {}).get('vol', 100)

        # ── mel 提取：先读更长音频留出 STFT 上下文余量，再裁掉边缘帧 ──
        # 这样 center=True 的首尾帧也有完整音频上下文，频谱更准确
        pad_context = int(round((n_fft // 2) / base_hop)) * base_hop  # 对齐到 hop 整数倍
        pad_front = min(pad_context, start_sample)
        pad_tail = min(pad_context, len(audio) - end_sample)

        # 注意：volume gain 已在 audio_seg 上应用，扩展部分不应用 gain（边缘数据，会被裁掉）
        audio_ext = audio[start_sample - pad_front : end_sample + pad_tail]
        mel_ext = self._audio_to_mel(audio_ext, base_hop, sr, n_fft, win_length)

        # 裁掉填充帧：pad_front/hop 和 pad_tail/hop 都是整数，帧对齐精确
        crop_front = pad_front // base_hop
        crop_tail = pad_tail // base_hop
        if crop_tail > 0:
            mel_full = mel_ext[:, crop_front:-crop_tail]
        else:
            mel_full = mel_ext[:, crop_front:]

        n_mels, n_frames = mel_full.shape

        if n_frames == 0:
            info['mel'] = np.empty((n_mels, 0))
            info['audio_seg'] = audio_seg
            info['consonant_frames'] = 0
            info['stretch_factor'] = 1.0
            return i

        # 辅音/元音帧数
        con_frames_orig = max(1, int((con_samples - base_hop) / base_hop) + 1) if con_samples > 0 else 0
        con_frames_orig = min(con_frames_orig, n_frames)
        vow_frames_orig = n_frames - con_frames_orig

        # stretch
        vel = info['Note_flags']['vel']
        stretch_factor = 2.0 ** ((100 - vel) / 100.0)
        p0_x = info['envelope']['p0']['x']
        p4_x = info['envelope']['p4']['x']
        stretched_preutter = Preutter_ms * stretch_factor

        # ── 总时间预算（仅音频内容，不含空白填充） ──
        total_budget_ms = p4_x + stretched_preutter
        total_budget_frames = max(int(total_budget_ms / self.ms_per_frame), 1)


        # 包络最左边界（取 p0/p1 中更左的那个）
        # p1_x = info['envelope']['p1']['x']
        # left_bound = min(p0_x, p1_x)
        left_bound = p0_x
        # 计算最左边界相对音频起点的位置
        # stretched_preutter 为正值（音频起点到中心的距离）
        # left_bound 为负值（中心到包络最左的距离）
        # pre_to_left_ms > 0 → 裁剪; < 0 → 补空白
        pre_to_left_ms = stretched_preutter + left_bound

        # 辅音帧数（拉伸后，浮点精度）
        target_con_frames = max(1, int(con_frames_orig * stretch_factor))
        target_con_frames = min(target_con_frames, total_budget_frames - 1)

        # 元音帧数 = 总帧数 - 辅音帧数（无额外 int 截断，保证总帧数精确匹配时间预算）
        target_vow_frames = total_budget_frames - target_con_frames
        total_frames = total_budget_frames

        # ── strt=1: 参考 He 标志 — 用 np.pad(mode='reflect') 做正反循环 ──
        # 原理：元音部分用 reflect padding 扩展，插值自然走正反循环
        # 注意：pad_frames = target - orig + extra(2帧)，让 vow_frames_orig > target_vow_frames
        # 这样 mapped 索引全部是分数→interp1d 做帧间混合→平滑过渡无咔哒声
        # 若 pad 到精确相等，vow 映射比 = 1.0，mapped 索引为整数→无插值→硬切
        strt = info.get('Note_flags', {}).get('strt', 0)
        if (strt == 1 and vow_frames_orig > 1
                and target_vow_frames > vow_frames_orig * 1.5):
            print(f"  循环: {info['phoneme_name']} 元音 {vow_frames_orig}→{target_vow_frames}帧")
            mel_vowel = mel_full[:, con_frames_orig:]  # 元音部分
            pad_extra = min(4, vow_frames_orig // 2)   # 额外多 pad 帧数，保证分数索引插值
            pad_frames = target_vow_frames - vow_frames_orig + pad_extra
            mel_vowel_ext = np.pad(mel_vowel, ((0, 0), (0, pad_frames)),
                                   mode='reflect')
            mel_full = np.concatenate(
                [mel_full[:, :con_frames_orig], mel_vowel_ext], axis=1)
            n_frames = mel_full.shape[1]
            vow_frames_orig = n_frames - con_frames_orig

        # 帧索引映射
        old_idx = np.arange(n_frames)
        new_idx = np.arange(total_frames, dtype=np.float64)

        con_mask = new_idx < target_con_frames
        vow_mask = ~con_mask
        mapped = np.empty(total_frames, dtype=np.float64)
        mapped[con_mask] = new_idx[con_mask] / stretch_factor
        if np.any(vow_mask) and vow_frames_orig > 0:
            vow_out_start = target_con_frames
            mapped[vow_mask] = con_frames_orig + (new_idx[vow_mask] - vow_out_start) * (
                vow_frames_orig / max(1, target_vow_frames))
        elif np.any(vow_mask):
            mapped[vow_mask] = con_frames_orig
        mapped = np.clip(mapped, 0, n_frames - 1)

        # ── 线性拉伸插值（与旧版一致） ──
        mel_out = interp1d(old_idx, mel_full, axis=1, kind='linear',
                           bounds_error=False, fill_value='extrapolate')(mapped)

        # ── strt=1: 抹平循环段音量起伏，代之以首→尾渐变 ──
        # reflect padding 导致循环处频谱反复，能量周期性波动。
        # 这里将元音部分每帧能量归一化到相同水平，再线性渐变回原始首→尾能量。
        if strt == 1 and target_vow_frames > 1 and mel_out.shape[1] >= target_con_frames + 2:
            vowel = mel_out[:, target_con_frames:]                      # (n_mels, T_vow)
            frame_energy = np.mean(np.exp(vowel), axis=0)               # (T_vow,), 线性域每帧平均能量
            frame_energy = np.maximum(frame_energy, 1e-12)              # 保护极小值
            # 目标能量：从首帧到末帧线性渐变，保持原始整体能量趋势
            target_e = np.linspace(frame_energy[0], frame_energy[-1], target_vow_frames)
            # mel - log(energy) + log(target) = log(exp(mel) / energy * target)
            mel_out[:, target_con_frames:] = (
                vowel - np.log(frame_energy)[np.newaxis, :]
                       + np.log(target_e)[np.newaxis, :]
            )
            print(f"  循环: {info['phoneme_name']} 元音 {vow_frames_orig}→{target_vow_frames}帧，能量归一化")

        # ── 左侧裁剪/补空白（基于包络最左边界 left_bound） ──
        # pre_to_left_ms > 0: 左边界在音频起点之后 → 裁剪前面多余帧
        # pre_to_left_ms < 0: 左边界在音频起点之前 → 前面补空白帧
        if pre_to_left_ms > 0:
            left_cut_frames = int(pre_to_left_ms / self.ms_per_frame)
            if left_cut_frames < mel_out.shape[1]:
                mel_out = mel_out[:, left_cut_frames:]
            else:
                mel_out = np.empty((n_mels, 0))
        elif pre_to_left_ms < 0:
            left_pad_frames = int(-pre_to_left_ms / self.ms_per_frame)
            blank = np.full((n_mels, left_pad_frames),
                            np.log(1e-5), dtype=mel_out.dtype)
            # 渐入：用 4 帧从空白平滑过渡到真实音频，避免 HiFi-GAN 解码硬切换噪声
            # 注意：使用逐频带 fade-in，保留频谱形状，避免平坦频谱产生咔哒声
            if mel_out.shape[1] > 0:
                fade_in_frames = min(4, left_pad_frames)
                first_frame = mel_out[:, 0]  # 保留各频带原始能量分布
                for t in range(fade_in_frames):
                    alpha = (t + 1) / (fade_in_frames + 1)
                    # 每频带独立淡入，保留频谱形状
                    blank[:, left_pad_frames - fade_in_frames + t] = np.log(
                        np.exp(first_frame) * alpha + 1e-10
                    )
            mel_out = np.concatenate([blank, mel_out], axis=1)

        # ── 首/尾音素淡入淡出已移至 engine.py 波形域统一处理 ──

        # ── P 参数：音量均衡 — 趋近于最大可能音量的一半（-6dB） ──
        P = info.get('Note_flags', {}).get('P', 0)
        if P > 0 and mel_out.shape[1] > 0:
            target_rms = 0.5  # 固定目标：音频最大音量的一半
            audio_start = max(0, int(-pre_to_left_ms / self.ms_per_frame) if pre_to_left_ms < 0 else 0)
            if mel_out.shape[1] > audio_start:
                mel_audio = mel_out[:, audio_start:]
                cur_rms = float(np.sqrt(np.mean(np.exp(mel_audio) ** 2)))
                if cur_rms > 1e-12:
                    blend = P / 100.0
                    target_rms_actual = cur_rms * (1 - blend) + target_rms * blend
                    scale = target_rms_actual / cur_rms
                    mel_out = mel_out + np.log(scale)
                    print(f"  P={P}: {info['phoneme_name']} "
                          f"cur={cur_rms:.4f} target={target_rms_actual:.4f} (x{scale:.4f})")

        info['mel'] = mel_out
        info['audio_seg'] = audio_seg
        info['consonant_frames'] = target_con_frames
        info['stretch_factor'] = stretch_factor

        print(f"完成: {info['phoneme_name']} | {os.path.basename(wav_path)} "
              f"O={offset_ms} C={consonant_ms} F={cutoff_ms} "
              f"vel={vel}(×{stretch_factor:.2f}) mel {mel_out.shape[1]}帧 (con={target_con_frames})")
        return i


    @staticmethod
    def _audio_to_mel(audio: np.ndarray, hop: int, sr: int,
                       n_fft: int, win_length: int) -> np.ndarray:
        """音频 → log-mel 谱。"""
        if len(audio) == 0:
            return np.empty((128, 0))
        # 确保单声道（有些 WAV 是立体声）
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # 使用 center=True 让 librosa 自动处理短信号（极短音素保护）
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft,
            hop_length=hop, win_length=win_length,
            n_mels=128, fmin=40, fmax=16000,
            center=True, power=1.0,
        )
        return dynamic_range_compression(mel)

    # ─── mel 调试图片（测试用） ───
    def _save_mel_debug_image(self, output_dir: str = "synthesis_pipeline/mel_debug"):
        """将每个音素的 mel 频谱保存为 PNG 图片，用于测试验证。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        for i, info in enumerate(self.phoneme_list):
            mel = info.get('mel')
            if mel is None or mel.shape[1] == 0:
                continue
            name = info.get('phoneme_name', f'phoneme_{i}')
            cons = info.get('consonant_frames', 0)
            stretch = info.get('stretch_factor', 1.0)

            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(mel, aspect='auto', origin='lower',
                           cmap='magma', interpolation='nearest')
            ax.axvline(x=cons - 0.5, color='cyan', linestyle='--',
                       linewidth=1, label=f'consonant={cons}')
            ax.set_title(f'{name}  (#{i})  |  con={cons}  '
                         f'total={mel.shape[1]}帧  '
                         f'stretch=×{stretch:.2f}')
            ax.set_xlabel('帧')
            ax.set_ylabel('mel 频带')
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
            cbar.set_label('log-mel')
            ax.legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            safe_name = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            fig.savefig(os.path.join(output_dir, f'{i:03d}_{safe_name}.png'),
                        dpi=150)
            plt.close(fig)

    # ─── 主入口 ───
    def cut_audio(self, max_workers: int = 4, save_mel_image: bool = False):
        """预处理交叉帧 → 多线程并行处理所有音素。

        Args:
            max_workers: 并行线程数。
            save_mel_image: 处理后是否保存每个音素的 mel 频谱图片到 mel_debug/。
        """
        # 交叉帧计算移至 hidden_splicer 内部用 round() 处理
        n = len(self.phoneme_list)
        if n <= 1:
            for i in range(n):
                self._process_single_phoneme(i)
            return

        with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
            futures = {executor.submit(self._process_single_phoneme, i): i
                       for i in range(n)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    import traceback
                    traceback.print_exc()
                    print(f"音素 {idx} 处理失败: {e}")

        if save_mel_image:
            self._save_mel_debug_image()

    # ─── 音量匹配 (phtp) ───
    def adjust_volume_by_phtp(self):
        ms_per_frame = self.ms_per_frame

        def _mel_energy_rms(seg):
            return float(np.sqrt(np.mean(np.exp(seg) ** 2)))

        for i in range(len(self.phoneme_list)):
            info = self.phoneme_list[i]
            phtp = info.get('Note_flags', {}).get('phtp', 0)
            if phtp == 0:
                continue
            mel_cur = info.get('mel')
            if mel_cur is None or mel_cur.shape[1] == 0:
                continue

            if phtp == 1 and i < len(self.phoneme_list) - 1:
                next_info = self.phoneme_list[i + 1]
                mel_next = next_info.get('mel')
                if mel_next is None or mel_next.shape[1] == 0:
                    continue
                p0_x, p1_x = next_info['envelope']['p0']['x'], next_info['envelope']['p1']['x']
                overlap_ms = abs(p0_x) - abs(p1_x) if p1_x < 0 else abs(p1_x) + abs(p0_x)
                if overlap_ms <= 0:
                    continue
                overlap_frames = round(overlap_ms / ms_per_frame)
                cur_tail, next_head = min(overlap_frames, mel_cur.shape[1]), min(overlap_frames, mel_next.shape[1])
                if cur_tail <= 0 or next_head <= 0:
                    continue
                rms_cur = _mel_energy_rms(mel_cur[:, -cur_tail:])
                rms_next = _mel_energy_rms(mel_next[:, :next_head])
                if rms_cur < 1e-12 or rms_next < 1e-12:
                    continue
                scale = rms_next / rms_cur
                mel_cur[:, :] = mel_cur + np.log(scale)
                print(f"  phtp=1: 音素 {i} ({info['phoneme_name']}) 跟随后音 ×{scale:.4f}")

            elif phtp == 2 and i > 0:
                prev_info = self.phoneme_list[i - 1]
                mel_prev = prev_info.get('mel')
                if mel_prev is None or mel_prev.shape[1] == 0:
                    continue
                p0_x, p1_x = info['envelope']['p0']['x'], info['envelope']['p1']['x']
                overlap_ms = abs(p0_x) - abs(p1_x) if p1_x < 0 else abs(p1_x) + abs(p0_x)
                if overlap_ms <= 0:
                    continue
                overlap_frames = round(overlap_ms / ms_per_frame)
                prev_tail, cur_head = int(min(overlap_frames, mel_prev.shape[1])), int(min(overlap_frames, mel_cur.shape[1]))
                if prev_tail <= 0 or cur_head <= 0:
                    continue
                rms_prev = _mel_energy_rms(mel_prev[:, -prev_tail:])
                rms_cur = _mel_energy_rms(mel_cur[:, :cur_head])
                if rms_prev < 1e-12 or rms_cur < 1e-12:
                    continue
                scale = rms_prev / rms_cur
                mel_cur[:, :] = mel_cur + np.log(scale)
                print(f"  phtp=2: 音素 {i} ({info['phoneme_name']}) 跟随前音 ×{scale:.4f}")

    # ─── 音素级 shft 频域偏移 ───
    def apply_dynamic_gen_to_mels(self):
        """对每个音素的 mel 频谱应用频域弯曲。

        优先读取音素自身的 Dynamic_parameter.genc（时变数组，音分），
        回退到 Note_flags.shft（常数值，音分）。
        范围 ±200，对应 ±2 半音（100 = 1 半音）。
        """
        applied = False
        for i in range(len(self.phoneme_list)):
            info = self.phoneme_list[i]
            mel = info.get('mel')
            if mel is None or mel.shape[1] == 0:
                continue

            n_mels = mel.shape[0]
            old_idx = np.arange(n_mels)

            # ── 方案1: 音素级 Dynamic_parameter.genc 时变数组 ──
            ph_dp = info.get('Dynamic_parameter', {})
            ph_genc = ph_dp.get('genc')
            use_genc = (ph_genc is not None
                        and isinstance(ph_genc, (list, np.ndarray))
                        and len(ph_genc) > 0)

            if use_genc:
                ph_genc_arr = np.array(ph_genc, dtype=np.float32)
                genc_interp = interp_to_len(ph_genc_arr, mel.shape[1])
                # genc_interp = np.clip(genc_interp, -200, 200)
                semitone_curve = genc_interp / 100.0

                mel_out = np.zeros_like(mel)
                for t in range(mel.shape[1]):
                    factor = 2.0 ** (semitone_curve[t] / 12.0)
                    if abs(factor - 1.0) < 0.001:
                        mel_out[:, t] = mel[:, t]
                    else:
                        new_idx = np.clip(old_idx / factor, 0, n_mels - 1)
                        mel_out[:, t] = np.interp(new_idx, old_idx, mel[:, t])
                info['mel'] = mel_out
                applied = True
                print(f"  genc[{i}]: {info['phoneme_name']} "
                      f"[{ph_genc_arr.min():.1f}, {ph_genc_arr.max():.1f}]")
                continue

            # ── 方案2: Note_flags.shft 常数值（回退） ──
            shft_val = info.get('Note_flags', {}).get('shft', 0)
            if shft_val == 0:
                continue
            shft_val = float(np.clip(shft_val, -200, 200))
            semitones = shft_val / 100.0
            factor = 2.0 ** (semitones / 12.0)
            if abs(factor - 1.0) < 0.001:
                continue
            new_idx = np.clip(old_idx / factor, 0, n_mels - 1)
            mel_warped = np.zeros_like(mel)
            for t in range(mel.shape[1]):
                mel_warped[:, t] = np.interp(new_idx, old_idx, mel[:, t])
            info['mel'] = mel_warped
            applied = True
            print(f"  shft={shft_val:.0f}: {info['phoneme_name']} "
                  f"频域偏移 ×{factor:.4f}")

        if applied:
            print("  音素级频域偏移完成")

    # ─── 波形域音区偏移（相位声码器）已删除 ───
