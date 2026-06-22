"""
FragmentMel — SPLC=1 音素片段处理（mel 域拼接专用）

与标准 Fragment 不同，此版本直接以 hop=512 工作，使用外部 mel_exc
进行自定义帧定位的 mel 提取，专用于 mel 域能量叠加拼接。
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from synthesis_pipeline.utils import read_audio, dynamic_range_compression, resample_array, interp_to_len


class FragmentMel:
    """解析 JSON 并处理音频片段（SPLC=1 mel 域拼接版）。"""

    def __init__(self, json_data: dict, mel_exc):
        self.sample_rate = 44100
        self.hop_length = 512
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
        self.warm = self._get_param(dp, ('warm', 'warmth'))
        self.hcmp = self._get_param(dp, ('hcmp',))

        self.mel_exc = mel_exc

    @staticmethod
    def _get_param(dp: dict, keys: tuple) -> np.ndarray:
        for k in keys:
            val = dp.get(k)
            if val is not None:
                return np.array(val, dtype=np.float32)
        return np.array([], dtype=np.float32)

    def _process_single_phoneme(self, i: int, starts, preutters, consonants, offsets, vfs, ends) -> int:
        """处理单个音素：读取音频 → mel 转换 → 时间拉伸。"""
        spm = self.sample_rate / 1000.0
        info = self.phoneme_list[i]
        oto = info['phoneme_oto']
        wav_path = oto['audio_file_path']

        if not os.path.exists(wav_path):
            print(f'音频文件不存在: {wav_path}')
            return i

        sr = self.sample_rate
        audio = read_audio(wav_path, sr)
        total_len = len(audio)

        oto_offset = oto['Offset'] * spm
        oto_consonant = (oto['Consonant'] + oto['Offset']) * spm
        oto_end = (total_len - oto['Cutoff'] * spm) if oto['Cutoff'] > 0 else (oto_offset + abs(oto['Cutoff'] * spm))
        oto_preutter = (oto['Preutter'] + oto['Offset']) * spm

        vf = vfs[i]
        consonant = consonants[i]
        offset = offsets[i]
        end = ends[i]
        sf = (end - consonant) / (oto_end - oto_consonant) if (oto_end - oto_consonant) > 0 else 1.0
        strt = info.get('Note_flags', {}).get('strt', 0)

        mel_offset = int((offset - self.hop_length / 2) // self.hop_length)
        mel_consonant = int((consonant - self.hop_length / 2) // self.hop_length)
        mel_end = int((end - self.hop_length / 2) // self.hop_length + 1)
        frame = int(mel_end - mel_offset + 1)
        con_frames = mel_end - mel_consonant + 1

        h_start = offset - (offset - self.hop_length / 2) % self.hop_length
        h_points = np.arange(frame) * self.hop_length + h_start

        def fn(x):
            if strt == 1:
                return np.where(x < consonant,
                                oto_consonant - (consonant - x) / vf,
                                oto_end - np.abs((x - consonant) % (2 * (oto_end - oto_consonant)) - (oto_end - oto_consonant)))
            else:
                return np.where(x < consonant,
                                oto_consonant - (consonant - x) / vf,
                                oto_consonant + (x - consonant) / sf)

        oto_h_points = fn(h_points)
        mel = self.mel_exc(audio, oto_h_points)
        mel = dynamic_range_compression(mel)

        # strt=1: 抹平循环段音量起伏
        if strt == 1 and con_frames < mel.shape[1]:
            vowel = mel[:, :-con_frames]
            frame_energy = np.mean(np.exp(vowel), axis=0)
            frame_energy = np.maximum(frame_energy, 1e-12)
            target_e = np.linspace(frame_energy[0], frame_energy[-1], len(frame_energy))
            mel[:, :-con_frames] = (
                vowel - np.log(frame_energy)[np.newaxis, :]
                       + np.log(target_e)[np.newaxis, :]
            )
            print(f"  循环: {info['phoneme_name']} 元音能量归一化")

        # P 参数：音量均衡
        P = info.get('Note_flags', {}).get('P', 0)
        if P > 0 and mel.shape[1] > 0:
            target_rms = 0.5
            cur_rms = float(np.sqrt(np.mean(np.exp(mel) ** 2)))
            if cur_rms > 1e-12:
                blend = P / 100.0
                target_rms_actual = cur_rms * (1 - blend) + target_rms * blend
                scale = target_rms_actual / cur_rms
                mel = mel + np.log(scale)
                print(f"  P={P}: {info['phoneme_name']} "
                      f"cur={cur_rms:.4f} target={target_rms_actual:.4f} (x{scale:.4f})")

        info['mel'] = mel
        info['audio_seg'] = audio
        info['mel_offset'] = mel_offset
        info['mel_end'] = mel_end
        info['preutter'] = preutters[i]
        info['h_points'] = h_points

        print(f"完成: {info['phoneme_name']} | {os.path.basename(wav_path)} "
              f"O={offset} C={consonant} F={end} "
              f"mel {mel.shape[1]}帧 (con={con_frames})")
        return i

    def calc_positions_and_ratios_ms(self, phoneme_list):
        """预先计算所有音素的时间位置。"""
        spm = self.sample_rate / 1000.0
        starts = []
        starts_ms = []
        preutters = []
        consonants = []
        offsets = []
        vfs = []
        ends = []

        for i, info in enumerate(phoneme_list):
            env = info["envelope"]
            p0 = env["p0"]["x"]
            p1 = env["p1"]["x"]
            p4 = env["p4"]["x"]
            ov = p1 - p0

            if i == 0:
                s = 0.0
            else:
                prev_env = phoneme_list[i - 1]["envelope"]
                prev_p0 = prev_env["p0"]["x"]
                prev_p4 = prev_env["p4"]["x"]
                prev_len = prev_p4 - prev_p0
                s = starts_ms[-1] + prev_len - ov

            preutter = s - p0

            oto = info['phoneme_oto']
            consonant_ms = oto['Consonant']
            preutter_ms = oto['Preutter']
            vel = info['Note_flags']['vel']

            vf = 2.0 ** ((100.0 - vel) / 100.0)
            consonant = preutter + (consonant_ms - preutter_ms) * vf
            offset = consonant - consonant_ms * vf
            end_pos = s + (p4 - p0)

            starts_ms.append(s)
            starts.append(s * spm)
            preutters.append(preutter * spm)
            consonants.append(consonant * spm)
            offsets.append(offset * spm)
            vfs.append(vf)
            ends.append(end_pos * spm)

        return starts, preutters, consonants, offsets, vfs, ends

    def cut_audio(self, max_workers: int = 1):
        """串行处理所有音素。"""
        starts, preutters, consonants, offsets, vfs, ends = self.calc_positions_and_ratios_ms(self.phoneme_list)
        for i in range(len(self.phoneme_list)):
            self._process_single_phoneme(i, starts, preutters, consonants, offsets, vfs, ends)

    def adjust_volume_by_phtp(self):
        """音量匹配 (phtp) — 同标准 Fragment。"""
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

    def apply_dynamic_gen_to_mels(self):
        """音素级 shft/genc 频域偏移。"""
        applied = False
        for i in range(len(self.phoneme_list)):
            info = self.phoneme_list[i]
            mel = info.get('mel')
            if mel is None or mel.shape[1] == 0:
                continue

            n_mels = mel.shape[0]
            old_idx = np.arange(n_mels)

            ph_dp = info.get('Dynamic_parameter', {})
            ph_genc = ph_dp.get('genc')
            use_genc = (ph_genc is not None
                        and isinstance(ph_genc, (list, np.ndarray))
                        and len(ph_genc) > 0)

            if use_genc:
                ph_genc_arr = np.array(ph_genc, dtype=np.float32)
                genc_interp = interp_to_len(ph_genc_arr, mel.shape[1])
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
            print(f"  shft={shft_val:.0f}: {info['phoneme_name']} 频域偏移 ×{factor:.4f}")

        if applied:
            print("  音素级频域偏移完成")
