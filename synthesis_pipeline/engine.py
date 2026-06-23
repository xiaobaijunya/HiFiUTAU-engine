"""
合成引擎 — 接收预加载的模型，执行完整合成管线。
"""
import time
import io
import numpy as np
import soundfile as sf

from synthesis_pipeline.fragment import Fragment
from synthesis_pipeline.fragment_mel import FragmentMel
from synthesis_pipeline.post_process import apply_hnsep_postprocess, apply_hnsep_postprocess_components
from synthesis_pipeline.growl import apply_growl
from synthesis_pipeline.tension_filter import apply_dynamic_lowcut
from synthesis_pipeline.warmth import apply_warmth_eq, apply_harmonic_compression
from synthesis_pipeline.utils import resample_array, interp_to_len, hnsep_separate


class SynthesisEngine:
    """语音合成引擎。

    模型在外部加载后注入，引擎本身不管理模型生命周期。
    可安全在多线程环境中复用（每次调用 synthesize() 创建独立的 Fragment）。

    Usage:
        engine = SynthesisEngine(splicer=my_splicer, hnsep_session=my_session)
        wav_bytes = engine.synthesize(json_data)
    """

    def __init__(self, splicer, hnsep_session=None, mel_exc=None):
        """
        Args:
            splicer:        HiddenSplicer 实例（已加载 ONNX 模型）
            hnsep_session:  HN-SEP ONNX 推理会话（可选）
            mel_exc:        PitchAndTimeAdjustableMelSpectrogram（SPLC=1 时使用）
        """
        self._splicer = splicer
        self._hnsep = hnsep_session
        self._mel_exc = mel_exc

    # ─── 属性（只读） ───
    @property
    def splicer(self):
        return self._splicer

    @property
    def hnsep_session(self):
        return self._hnsep

    # ─── 合成 ───
    def synthesize(self, json_data: dict, *,
                   test: bool = False,
                   max_workers: int = 2) -> bytes:
        t_start = time.time()

        # 检查是否有 SPLC=1 音素，决定使用哪种管线
        # 默认 SPLC=1（未设置时按 mel 域能量叠加拼接处理）
        use_mel_pipeline = any(
            info.get('Note_flags', {}).get('splc', 1) == 1
            for info in json_data['phoneme_list'].values()
        )

        if use_mel_pipeline and self._mel_exc is not None:
            return self._synthesize_mel(json_data, test, max_workers, t_start)
        else:
            return self._synthesize_feat(json_data, test, max_workers, t_start)

    def _synthesize_feat(self, json_data: dict, test: bool, max_workers: int, t_start: float) -> bytes:
        """SPLC=0: 标准 feat 域隐空间拼接管线。"""
        frag = Fragment(json_data)
        print(f"输出: {frag.out_wav} | 时长: {frag.wav_dur}ms | "
              f"音素数: {len(frag.phoneme_list)}")
        print(f"动态参数: tension={len(frag.tension)}帧, "
              f"breath={len(frag.breath)}帧, "
              f"voicing={len(frag.voicing)}帧, "
              f"growl={len(frag.growl)}帧, "
              f"brel={len(frag.brel)}帧, breh={len(frag.breh)}帧, "
              f"warm={len(frag.warm)}帧, hcmp={len(frag.hcmp)}帧")

        frag.cut_audio(max_workers=max_workers)
        frag.adjust_volume_by_phtp()
        frag.apply_dynamic_gen_to_mels()

        for info in frag.phoneme_list:
            vol = info.get('Note_flags', {}).get('vol', 100)
            gain = vol / 100.0
            if abs(gain - 1.0) > 1e-6 and info.get('mel') is not None and info['mel'].shape[1] > 0:
                info['mel'] = info['mel'] + np.log(gain)
                print(f"  VOL: {info['phoneme_name']} x{gain:.4f}")

        f0 = np.array(frag.pit, dtype=np.float32)
        target_hop = 512
        print(f"重采样 F0: {len(f0)} 帧 -> ", end="")
        f0 = resample_array(f0, frag.Dynamic_hop, target_hop)
        print(f"{len(f0)} 帧")

        print("隐空间混合拼接 (feat 域)...")
        wav = self._splicer.splice_and_synthesize(
            frag.phoneme_list, frag.ms_per_frame, frag.hop_length, f0
        )

        return self._postprocess(wav, frag, t_start, test)

    def _synthesize_mel(self, json_data: dict, test: bool, max_workers: int, t_start: float) -> bytes:
        """SPLC=1: mel 域能量叠加拼接管线。"""
        frag = FragmentMel(json_data, self._mel_exc)
        print(f"输出: {frag.out_wav} | 时长: {frag.wav_dur}ms | "
              f"音素数: {len(frag.phoneme_list)}")
        print(f"动态参数: tension={len(frag.tension)}帧, "
              f"breath={len(frag.breath)}帧, "
              f"voicing={len(frag.voicing)}帧, "
              f"growl={len(frag.growl)}帧, "
              f"brel={len(frag.brel)}帧, breh={len(frag.breh)}帧, "
              f"warm={len(frag.warm)}帧, hcmp={len(frag.hcmp)}帧")
        print("拼接模式: mel 域能量叠加 (SPLC=1)")

        frag.cut_audio(max_workers=max_workers)
        frag.adjust_volume_by_phtp()
        frag.apply_dynamic_gen_to_mels()

        for info in frag.phoneme_list:
            vol = info.get('Note_flags', {}).get('vol', 100)
            gain = vol / 100.0
            if abs(gain - 1.0) > 1e-6 and info.get('mel') is not None and info['mel'].shape[1] > 0:
                info['mel'] = info['mel'] + np.log(gain)
                print(f"  VOL: {info['phoneme_name']} x{gain:.4f}")

        f0 = np.array(frag.pit, dtype=np.float32)
        target_hop = 512
        print(f"重采样 F0: {len(f0)} 帧 -> ", end="")
        f0 = resample_array(f0, frag.Dynamic_hop, target_hop)
        print(f"{len(f0)} 帧")

        wav = self._splicer.splice_and_synthesize_mel(frag.phoneme_list, f0)

        return self._postprocess(wav, frag, t_start, test)

    def _postprocess(self, wav, frag, t_start, test):
        """HN-SEP 后处理 + WAV 输出（两管线共用）。"""

        # ── 计算补帧 ──
        front_dh = round(self._splicer.front_pad_frames * self._splicer.model_hop / frag.Dynamic_hop)
        tail_dh  = round(self._splicer.tail_pad_frames * self._splicer.model_hop / frag.Dynamic_hop)

        def _pad(arr, f, t):
            if len(arr) == 0:
                return arr
            fp = np.full(f, arr[0], dtype=arr.dtype) if f > 0 else np.array([], dtype=arr.dtype)
            tp = np.full(t, arr[-1], dtype=arr.dtype) if t > 0 else np.array([], dtype=arr.dtype)
            return np.concatenate([fp, arr, tp])

        # ── 6. HN-SEP 统一管线（真正的一次分离，多次处理） ──
        # 收集所有依赖 HN-SEP 的参数
        _need_hnsep_breath = len(frag.breath) > 0 and not np.allclose(frag.breath, 0, atol=0.5)
        _need_hnsep_tension = len(frag.tension) > 0 and not np.allclose(frag.tension, 0, atol=0.5)
        _need_hnsep_voicing = len(frag.voicing) > 0 and not np.allclose(frag.voicing, 100, rtol=0.05)
        _need_hnsep_brel = len(frag.brel) > 0 and not np.allclose(frag.brel, 0, atol=0.5)
        _need_hnsep_breh = len(frag.breh) > 0 and not np.allclose(frag.breh, 0, atol=0.5)
        _need_hnsep_hcmp = len(frag.hcmp) > 0 and not np.allclose(frag.hcmp, 0, atol=0.5)
        _need_hnsep_warm = len(frag.warm) > 0 and not np.allclose(frag.warm, 0, atol=0.5)

        _need_hnsep_legacy = (_need_hnsep_breath or _need_hnsep_tension or _need_hnsep_voicing
                              or _need_hnsep_brel or _need_hnsep_breh)

        if _need_hnsep_legacy or _need_hnsep_hcmp or _need_hnsep_warm:
            if self._hnsep is None:
                raise RuntimeError(
                    "当前合成使用了依赖 HN-SEP 的参数 (breath/tension/voicing/brel/breh/warm/hcmp)，"
                    "但 HN-SEP 模型未加载。请确保 hnsep 模型路径正确。"
                )

            print("HN-SEP 管线: 一次分离谐波/噪声...")
            harmonic, noise = hnsep_separate(wav, self._hnsep)

            # ── 6a. breath/tension/voicing/brel/breh ──
            #     直接操作已分离的分量，不再内部重新分离
            if _need_hnsep_legacy:
                _pad_b = _pad(frag.breath, front_dh, tail_dh)
                _pad_t = _pad(frag.tension, front_dh, tail_dh)
                _pad_v = _pad(frag.voicing, front_dh, tail_dh)
                _pad_f0 = _pad(frag.pit, front_dh, tail_dh)
                _pad_brel = _pad(frag.brel, front_dh, tail_dh)
                _pad_breh = _pad(frag.breh, front_dh, tail_dh)

                harmonic, noise = apply_hnsep_postprocess_components(
                    harmonic, noise, _pad_b, _pad_t, _pad_v, frag.sample_rate,
                    f0_curve=_pad_f0, brel_array=_pad_brel, breh_array=_pad_breh,
                )

            # ── 6b. 温暖度 EQ — 仅作用于谐波 ──
            if _need_hnsep_warm:
                warm_val = float(np.mean(frag.warm))
                print(f"  温暖度 EQ: warmth={warm_val:.1f}")
                harmonic = apply_warmth_eq(
                    harmonic, warm_val, frag.sample_rate, hnsep_session=None
                )

            # ── 6c. 谐波压缩（hcmp）— 放在处理链最后，控制最终谐波音量 ──
            if _need_hnsep_hcmp:
                hcmp_val = float(np.mean(frag.hcmp))
                print(f"  谐波压缩: hcmp={hcmp_val:.1f}")
                harmonic = apply_harmonic_compression(
                    harmonic, hcmp_val, frag.sample_rate, hnsep_session=None
                )

            # ── 6d. 混合 ──
            wav = harmonic + noise
            print("  HN-SEP 管线处理完成")

        # ── 7. 低切（F0 跟随 Butterworth 高通） ──
        if len(frag.lowcut) > 0 and not np.allclose(frag.lowcut, 0, atol=0.5):
            print("低切...")
            wav = apply_dynamic_lowcut(
                wav,
                interp_to_len(_pad(frag.lowcut, front_dh, tail_dh), len(wav)),
                frag.sample_rate,
                f0_curve=_pad(frag.pit, front_dh, tail_dh),
            )

        # ── 8. 咆哮效果（所有参数之后，最后一步） ──
        # 传入原始 F0 让咆哮频率跟随音高（也同步补齐补帧）
        if len(frag.growl) > 0:
            wav = apply_growl(wav,
                              _pad(frag.growl, front_dh, tail_dh),
                              frag.sample_rate,
                              f0=_pad(frag.pit, front_dh, tail_dh),
                              f0_hop=frag.Dynamic_hop)

        # ── 11. 统一裁剪首尾补帧 ──
        # HiFi-GAN 和 HN-SEP 都已受益于上下文，现在裁掉
        front_trim = self._splicer.front_pad_frames * self._splicer.model_hop
        tail_trim = self._splicer.tail_pad_frames * self._splicer.model_hop
        if front_trim > 0 and len(wav) > front_trim:
            wav = wav[front_trim:]
        if tail_trim > 0 and len(wav) > tail_trim:
            wav = wav[:-tail_trim]

        # ── 12. 输出前：按首音素 p0→p1 / 尾音素 p3→p4 包络淡入淡出 ──
        first_env = frag.phoneme_list[0]['envelope']
        last_env = frag.phoneme_list[-1]['envelope']
        sr = frag.sample_rate

        # 淡入（p0→p1）
        # 从 1e-6 而非 p0y/100 开始，避免 p0y=0 时首采样精确归零
        p0x, p0y = first_env['p0']['x'], first_env['p0']['y']
        p1x, p1y = first_env['p1']['x'], first_env['p1']['y']
        fade_in_len = max(0, round((p1x - p0x) / 1000 * sr))
        if fade_in_len > 1 and len(wav) > fade_in_len and p0y < p1y:
            start_gain = max(p0y / 100.0, 1e-6)
            gain_in = np.linspace(start_gain, p1y / 100.0, fade_in_len)
            wav[:fade_in_len] *= gain_in

        # 淡出（p3→p4）
        p3x, p3y = last_env['p3']['x'], last_env['p3']['y']
        p4x, p4y = last_env['p4']['x'], last_env['p4']['y']
        fade_out_len = max(0, round((p4x - p3x) / 1000 * sr))
        if fade_out_len > 1 and len(wav) > fade_out_len and p3y > p4y:
            gain_out = np.linspace(p3y / 100.0, p4y / 100.0, fade_out_len)
            wav[-fade_out_len:] *= gain_out

        # ── 13. 输出 ──
        if test:
            sf.write('./test_hidden_splice.wav', wav, 44100, 'PCM_16')
            print(f'保存: ./test_hidden_splice.wav ({len(wav)} 采样)')

        buf = io.BytesIO()
        sf.write(buf, wav, 44100, 'PCM_16', format='WAV')
        result = buf.getvalue()

        print(f"[OK] 合成耗时: {time.time() - t_start:.3f}s")
        return result
