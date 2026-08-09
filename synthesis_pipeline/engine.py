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


# ============================================================================
# WAV 编解码工具
# ============================================================================

def wav_bytes_to_samples(data: bytes) -> np.ndarray:
    """WAV 字节 → float32 单声道采样。

    注意：soundfile.read 返回 (data, samplerate) 元组，这里只取数据。
    """
    return sf.read(io.BytesIO(data), dtype='float32', always_2d=False)[0]


def samples_to_wav_bytes(wav: np.ndarray, sample_rate: int = 44100) -> bytes:
    """float32 采样 → PCM16 WAV 字节。"""
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, 'PCM_16', format='WAV')
    return buf.getvalue()


def load_splicer_config(config_path: str):
    """轻量 splicer 配置（不加载任何 ONNX/模型）。

    hnsep/post 阶段只需要补帧常数与采样率，无需加载完整的 part1/part2 模型，
    直接读取 config.json 即可，节省内存与启动时间。

    Returns:
        具有 model_hop / sample_rate / front_pad_frames / tail_pad_frames 属性的对象
    """
    import json as _json

    with open(config_path, encoding='utf-8') as f:
        cfg = _json.load(f)

    class _SplicerConfig:
        pass

    c = _SplicerConfig()
    c.model_hop = int(cfg['hop_size'])
    c.sample_rate = int(cfg['sampling_rate'])
    c.front_pad_frames = 6  # 与 BaseSplicer 中一致
    c.tail_pad_frames = 4   # 与 BaseSplicer 中一致
    return c


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

    # ─── 完整合成（旧接口，保持不变） ───
    def synthesize(self, json_data: dict, *,
                   test: bool = False,
                   max_workers: int = 2) -> bytes:
        t_start = time.time()

        frag, is_mel = self._build_fragment(json_data)
        frag, f0 = self._prepare_mel(frag, max_workers)
        wav = self._synthesize_hifigan(frag, f0, is_mel)

        return self._postprocess(wav, frag, t_start, test)

    # ════════════════════════════════════════════════════════════════
    #  分段合成 API（HiFiUTAU Local 渲染器）
    # ════════════════════════════════════════════════════════════════

    def synthesize_mel(self, json_data: dict, *,
                       max_workers: int = 4,
                       test: bool = False):
        """分段1: mel 拼接 + 变调(genc) + HiFi-GAN 合成。

        Args:
            json_data:   完整音素 JSON（含 oto/f0/genc/vel/vol/phtp/envelope/splc）
            max_workers: cut_audio 并行线程数
            test:        是否写出测试 WAV

        Returns:
            (wav_bytes, written=False)：引擎不落盘，written 恒为 False（统一回传）
        """
        t_start = time.time()

        frag, is_mel = self._build_fragment(json_data)
        frag, f0 = self._prepare_mel(frag, max_workers)
        wav = self._synthesize_hifigan(frag, f0, is_mel)

        if test:
            sf.write('./test_hifigan.wav', wav, 44100, 'PCM_16')
            print(f'保存: ./test_hifigan.wav ({len(wav)} 采样)')

        wav_bytes = samples_to_wav_bytes(wav)
        print(f"[OK] mel 合成耗时: {time.time()-t_start:.3f}s")
        return wav_bytes, False

    def synthesize_hnsep(self, wav_bytes: bytes):
        """分段2: HN-SEP 气声/谐波分离。

        Args:
            wav_bytes: hifigan 输出 wav 字节（带补帧）

        Returns:
            (harmonic_bytes, noise_bytes, False, False)：引擎不落盘，统一回传
        """
        t_start = time.time()
        if self._hnsep is None:
            raise RuntimeError(
                "当前合成使用了依赖 HN-SEP 的参数 (breath/tension/voicing/brel/breh/warm/hcmp)，"
                "但 HN-SEP 模型未加载。请确保 hnsep 模型路径正确。"
            )

        wav = wav_bytes_to_samples(wav_bytes)
        print("HN-SEP 管线: 分离谐波/噪声...")
        harmonic, noise = hnsep_separate(wav, self._hnsep)

        hb = samples_to_wav_bytes(harmonic)
        nb = samples_to_wav_bytes(noise)
        print(f"[OK] hnsep 分离耗时: {time.time()-t_start:.3f}s")
        return hb, nb, False, False

    def synthesize_post(self, json_data: dict, *,
                        wav_bytes: bytes | None = None,
                        harmonic_bytes: bytes | None = None,
                        noise_bytes: bytes | None = None,
                        max_workers: int = 4,
                        test: bool = False):
        """分段3: 参数应用（HN-SEP 参数 / 温暖度 / hcmp / 低切 / 咆哮 / 裁剪 / 包络）。

        两种输入模式:
          A. 提供 harmonic_bytes + noise_bytes（OpenUtau 已有 hnsep 缓存）→ 直接处理分量
          B. 仅提供 wav_bytes（无需 hnsep，或引擎内部自行分离）→ 完整后处理

        Args:
            json_data:      完整音素 JSON（用于解析参数/包络/F0）
            wav_bytes:      hifigan wav 字节（模式 B）
            harmonic_bytes: 谐波 wav 字节（模式 A）
            noise_bytes:    气声 wav 字节（模式 A）
            max_workers:    保留参数（post 不需要 cut_audio）
            test:           是否写出测试 WAV

        Returns:
            (final_bytes, written=False)：引擎不落盘，统一回传
        """
        t_start = time.time()

        # 仅解析参数/包络/F0（不执行 cut_audio）
        frag, _ = self._build_fragment(json_data)

        if harmonic_bytes is not None and noise_bytes is not None:
            # 模式 A: 直接处理已分离的分量
            harmonic = wav_bytes_to_samples(harmonic_bytes)
            noise = wav_bytes_to_samples(noise_bytes)
            final_bytes = self._postprocess(None, frag, t_start, test,
                                            harmonic=harmonic, noise=noise)
        elif wav_bytes is not None:
            # 模式 B: 从完整 wav 后处理（内部按需分离）
            wav = wav_bytes_to_samples(wav_bytes)
            final_bytes = self._postprocess(wav, frag, t_start, test)
        else:
            raise ValueError("synthesize_post 必须提供 wav_bytes 或 (harmonic_bytes, noise_bytes)")

        print(f"[OK] 参数应用耗时: {time.time()-t_start:.3f}s")
        return final_bytes, False

    # ─── 公共管线片段 ────────────────────────────────────

    def _build_fragment(self, json_data: dict):
        """创建 Fragment（根据 SPLC 选择 mel/feat 管线）。"""
        use_mel_pipeline = any(
            info.get('Note_flags', {}).get('splc', 1) == 1
            for info in json_data['phoneme_list'].values()
        )

        if use_mel_pipeline and self._mel_exc is not None:
            return FragmentMel(json_data, self._mel_exc), True
        else:
            return Fragment(json_data), False

    def _prepare_mel(self, frag, max_workers: int):
        """HiFi-GAN 之前的全部处理：切音频→mel→phtp→genc→VOL→F0。

        Returns:
            (frag, f0)
        """
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
        return frag, f0

    def _synthesize_hifigan(self, frag, f0: np.ndarray, is_mel_pipeline: bool):
        """HiFi-GAN 合成（feat 域拼接 或 mel 域能量叠加）。"""
        if is_mel_pipeline:
            print("拼接模式: mel 域能量叠加 (SPLC=1)")
            return self._splicer.splice_and_synthesize_mel(frag.phoneme_list, f0)
        print("隐空间混合拼接 (feat 域)...")
        return self._splicer.splice_and_synthesize(
            frag.phoneme_list, frag.ms_per_frame, frag.hop_length, f0
        )

    def _postprocess(self, wav, frag, t_start, test,
                     harmonic=None, noise=None):
        """HN-SEP 后处理 + 参数应用 + 输出（feat/mel 两管线共用）。

        Args:
            wav:       hifigan 输出波形。harmonic/noise 均非 None 时可为 None
                       （直接使用已分离的分量，不再内部重新分离）。
            harmonic:  预分离的谐波分量（可选）
            noise:     预分离的气声/噪声分量（可选）
        """
        # 若提供已分离分量，则以其混合作为基波
        if harmonic is not None and noise is not None:
            wav = harmonic + noise

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
            if harmonic is None or noise is None:
                # 需要内部重新分离 → 必须加载 hnsep 模型
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
