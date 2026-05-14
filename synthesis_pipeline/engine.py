"""
合成引擎 — 接收预加载的模型，执行完整合成管线。
"""
import time
import io
import numpy as np
import soundfile as sf

from synthesis_pipeline.fragment import Fragment
from synthesis_pipeline.post_process import apply_hnsep_postprocess
from synthesis_pipeline.growl import apply_growl
from synthesis_pipeline.tension_filter import apply_dynamic_lowcut
from synthesis_pipeline.utils import resample_array, interp_to_len


class SynthesisEngine:
    """语音合成引擎。

    模型在外部加载后注入，引擎本身不管理模型生命周期。
    可安全在多线程环境中复用（每次调用 synthesize() 创建独立的 Fragment）。

    Usage:
        engine = SynthesisEngine(splicer=my_splicer, hnsep_session=my_session)
        wav_bytes = engine.synthesize(json_data)
    """

    def __init__(self, splicer, hnsep_session=None):
        """
        Args:
            splicer:        HiddenSplicer 实例（已加载 ONNX 模型）
            hnsep_session:  HN-SEP ONNX 推理会话（可选）
        """
        self._splicer = splicer
        self._hnsep = hnsep_session

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
        """执行完整合成管线。

        Args:
            json_data:   OpenUTAU JSON 数据
            test:        是否同时写出测试 WAV 文件
            max_workers: cut_audio 并行线程数

        Returns:
            WAV 格式的音频 bytes
        """
        t_start = time.time()

        # ── 1. Fragment 初始化 ──
        frag = Fragment(json_data)
        print(f"输出: {frag.out_wav} | 时长: {frag.wav_dur}ms | "
              f"音素数: {len(frag.phoneme_list)}")
        print(f"动态参数: tension={len(frag.tension)}帧, "
              f"breath={len(frag.breath)}帧, "
              f"voicing={len(frag.voicing)}帧, "
              f"growl={len(frag.growl)}帧, "
              f"brel={len(frag.brel)}帧, breh={len(frag.breh)}帧")

        # ── 2. 音频切割 + mel（多线程） ──
        frag.cut_audio(max_workers=max_workers)

        # ── 3. 音量匹配 + gen ──
        frag.adjust_volume_by_phtp()
        frag.apply_dynamic_gen_to_mels()

        # ── 3b. VOL（放在 phtp→gen 之后，最后应用到 mel） ──
        for info in frag.phoneme_list:
            vol = info.get('Note_flags', {}).get('vol', 100)
            gain = vol / 100.0
            if abs(gain - 1.0) > 1e-6 and info.get('mel') is not None and info['mel'].shape[1] > 0:
                info['mel'] = info['mel'] + np.log(gain)
                print(f"  VOL: {info['phoneme_name']} x{gain:.4f}")

        # ── 4. F0 ──
        f0 = np.array(frag.pit, dtype=np.float32)
        target_hop = 512
        print(f"重采样 F0: {len(f0)} 帧 -> ", end="")
        f0 = resample_array(f0, frag.Dynamic_hop, target_hop)
        print(f"{len(f0)} 帧")

        # ── 5. 隐空间拼接 + 合成 ──
        # 检查是否有音素使用了 splc=1 标志（mel 域能量拼接）
        use_mel_crossfade = any(
            info.get('Note_flags', {}).get('splc', 0) == 1
            for info in frag.phoneme_list
        )
        if use_mel_crossfade:
            print("隐空间混合拼接 (混合模式: mel 域+feat 域)...")
            wav = self._splicer.splice_and_synthesize_mixed(
                frag.phoneme_list, frag.ms_per_frame, frag.hop_length, f0
            )
        else:
            print("隐空间混合拼接...")
            wav = self._splicer.splice_and_synthesize(
                frag.phoneme_list, frag.ms_per_frame, frag.hop_length, f0
            )

        # ── 6. HN-SEP 后处理 ──
        # 波形已包含首尾补帧，动态参数需同步补齐再传入
        front_dh = round(self._splicer.front_pad_frames * self._splicer.model_hop / frag.Dynamic_hop)
        tail_dh  = round(self._splicer.tail_pad_frames * self._splicer.model_hop / frag.Dynamic_hop)

        def _pad(arr, f, t):
            if len(arr) == 0:
                return arr
            fp = np.full(f, arr[0], dtype=arr.dtype) if f > 0 else np.array([], dtype=arr.dtype)
            tp = np.full(t, arr[-1], dtype=arr.dtype) if t > 0 else np.array([], dtype=arr.dtype)
            return np.concatenate([fp, arr, tp])

        if self._hnsep is not None:
            wav = apply_hnsep_postprocess(
                wav,
                _pad(frag.breath, front_dh, tail_dh),
                _pad(frag.tension, front_dh, tail_dh),
                _pad(frag.voicing, front_dh, tail_dh),
                frag.sample_rate, self._hnsep,
                f0_curve=_pad(frag.pit, front_dh, tail_dh),
                brel_array=_pad(frag.brel, front_dh, tail_dh),
                breh_array=_pad(frag.breh, front_dh, tail_dh),
            )

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

        # ── 9. 统一裁剪首尾补帧 ──
        # HiFi-GAN 和 HN-SEP 都已受益于上下文，现在裁掉
        front_trim = self._splicer.front_pad_frames * self._splicer.model_hop
        tail_trim = self._splicer.tail_pad_frames * self._splicer.model_hop
        if front_trim > 0 and len(wav) > front_trim:
            wav = wav[front_trim:]
        if tail_trim > 0 and len(wav) > tail_trim:
            wav = wav[:-tail_trim]

        # ── 9. 输出前：按首音素 p0→p1 / 尾音素 p3→p4 包络淡入淡出 ──
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

        # ── 10. 输出 ──
        if test:
            sf.write('./test_hidden_splice.wav', wav, 44100, 'PCM_16')
            print(f'保存: ./test_hidden_splice.wav ({len(wav)} 采样)')

        buf = io.BytesIO()
        sf.write(buf, wav, 44100, 'PCM_16', format='WAV')
        result = buf.getvalue()

        print(f"[OK] 合成耗时: {time.time() - t_start:.3f}s")
        return result
