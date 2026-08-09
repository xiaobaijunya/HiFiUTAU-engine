"""
hifiutau-engine ONNX 合成入口 — 模型加载 + 合成调度。

模型在此文件中加载&缓存，合成逻辑委托给 synthesis_pipeline。
"""
import json
import os
import onnxruntime

from tools.hidden_splicer import HiddenSplicer
from synthesis_pipeline import SynthesisEngine
from synthesis_pipeline.engine import load_splicer_config
from util.wav2mel import PitchAndTimeAdjustableMelSpectrogram


# ============================================================================
# 全局模型缓存
# ============================================================================

_onnx_splicer: HiddenSplicer | None = None
_onnx_splicer_config: tuple | None = None
_hnsep_session = None
_splicer_config = None


def get_splicer(part1_onnx: str, part2_onnx: str,
                config_path: str, device: str = 'cpu',
                infer_threads: int = 1) -> HiddenSplicer:
    """获取缓存的 HiddenSplicer，首次加载后复用。

    注：infer_threads 仅在首次加载时生效（缓存命中后忽略，避免重复加载）。
    """
    global _onnx_splicer, _onnx_splicer_config
    config_key = (part1_onnx, part2_onnx, config_path, device)
    if _onnx_splicer is None or _onnx_splicer_config != config_key:
        print("[缓存] 加载 HiddenSplicer ONNX 模型...")
        _onnx_splicer = HiddenSplicer(part1_onnx, part2_onnx,
                                       config_path, device=device,
                                       infer_threads=infer_threads)
        _onnx_splicer_config = config_key
        print("[缓存] HiddenSplicer 就绪")
    return _onnx_splicer


def get_hnsep_session():
    """获取 HN-SEP 全局会话。"""
    return _hnsep_session


def get_splicer_config():
    """轻量 splicer 配置（不加载 ONNX），供 hnsep/post 阶段使用。"""
    global _splicer_config
    if _splicer_config is None:
        onnx_dir = r"exported_onnx_v2"
        _splicer_config = load_splicer_config(
            os.path.join(onnx_dir, "config.json"))
    return _splicer_config


# ═══════════════════════════════════════════════════════════════
#  预加载（可独立预加载 splicer / hnsep）
# ═══════════════════════════════════════════════════════════════

def preload_splicer(device: str = 'dml', infer_threads: int = 1):
    """预加载 HiddenSplicer (part1/part2)。"""
    onnx_dir = r"exported_onnx_v2"
    get_splicer(
        os.path.join(onnx_dir, "part1.onnx"),
        os.path.join(onnx_dir, "part2.onnx"),
        os.path.join(onnx_dir, "config.json"),
        device,
        infer_threads=infer_threads,
    )


def preload_hnsep(device: str = 'dml', infer_threads: int = 1):
    """预加载 HN-SEP 模型。HN-SEP 用 pt2 新模型（spec→mask），支持 DML。"""
    global _hnsep_session
    try:
        hnsep_path = os.path.join(
            "hnsep_onnx", "hnsep_VR_44.1k_hop512_2024.05.pt2.onnx")
        if not os.path.exists(hnsep_path):
            hnsep_path = os.path.join(
                "hnsep_onnx", "hnsep_VR_44.1k_hop512_2024.05.onnx")

        dl = device.lower()
        if dl in ('dml', 'directml'):
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = max(1, int(infer_threads))

        print(f"加载 HN-SEP ONNX 模型: {hnsep_path}")
        _hnsep_session = onnxruntime.InferenceSession(
            hnsep_path, providers=providers, sess_options=so)
        print(f'HN-SEP ONNX 模型已加载, providers: {_hnsep_session.get_providers()}, '
              f'intra_op_threads={so.intra_op_num_threads}')
    except Exception as e:
        print(f"[警告] HN-SEP 预加载失败: {e}")


def preload_all(device: str = 'dml', infer_threads: int = 1):
    """预加载全部模型（splicer + hnsep）。"""
    import os  # 本地导入，避免 PyInstaller 环境下模块级 import 作用域问题
    preload_splicer(device, infer_threads=infer_threads)
    preload_hnsep(device, infer_threads=infer_threads)
    print("[OK] 全部模型就绪")


# ============================================================================
# 合成
# ============================================================================

def synthesize_audio(json_data: dict, *, test: bool = False,
                     max_workers: int = 4, device: str = 'dml') -> bytes:
    """执行完整合成管线。模型从缓存获取，无需每次传入。

    Args:
        json_data:   OpenUTAU JSON 数据
        test:        是否写出测试 WAV
        max_workers: cut_audio 并行线程数
        device:      'dml' (DirectML, 默认), 'cuda', 或 'cpu'

    Returns:
        WAV bytes
    """
    model_dir = r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    onnx_dir = r"exported_onnx_v2"

    splicer = get_splicer(
        os.path.join(onnx_dir, "part1.onnx"),
        os.path.join(onnx_dir, "part2.onnx"),
        os.path.join(onnx_dir, "config.json"),
        device,
    )
    hnsep = get_hnsep_session()

    mel_exc = PitchAndTimeAdjustableMelSpectrogram()
    engine = SynthesisEngine(splicer=splicer, hnsep_session=hnsep, mel_exc=mel_exc)
    return engine.synthesize(json_data, test=test, max_workers=max_workers)


# ============================================================================
# 分段合成（HiFiUTAU Local 渲染器）
# ============================================================================


def synthesize_mel(json_data: dict, *, max_workers: int = 4,
                   test: bool = False, device: str = 'dml'):
    """分段1: mel 拼接 + 变调 + HiFi-GAN。返回 (wav_bytes, written=False)。"""
    model_dir = r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    onnx_dir = r"exported_onnx_v2"

    splicer = get_splicer(
        os.path.join(onnx_dir, "part1.onnx"),
        os.path.join(onnx_dir, "part2.onnx"),
        os.path.join(onnx_dir, "config.json"),
        device,
    )
    mel_exc = PitchAndTimeAdjustableMelSpectrogram()
    engine = SynthesisEngine(splicer=splicer, hnsep_session=None, mel_exc=mel_exc)
    return engine.synthesize_mel(
        json_data, max_workers=max_workers, test=test)


def synthesize_hnsep(wav_bytes: bytes, *, device: str = 'dml'):
    """分段2: HN-SEP 气声/谐波分离。返回 (harmonic, noise, False, False)。"""
    hnsep = get_hnsep_session()
    engine = SynthesisEngine(splicer=get_splicer_config(),
                             hnsep_session=hnsep, mel_exc=None)
    return engine.synthesize_hnsep(wav_bytes)


def synthesize_post(json_data: dict, *, wav_bytes: bytes | None = None,
                    harmonic_bytes: bytes | None = None,
                    noise_bytes: bytes | None = None,
                    max_workers: int = 4, test: bool = False,
                    device: str = 'dml'):
    """分段3: 参数应用。返回 (final_bytes, written=False)。

    纯 CPU 轻量计算（numpy/scipy 滤波），不加载任何模型。
    """
    engine = SynthesisEngine(splicer=get_splicer_config(),
                             hnsep_session=None, mel_exc=None)
    return engine.synthesize_post(
        json_data, wav_bytes=wav_bytes,
        harmonic_bytes=harmonic_bytes, noise_bytes=noise_bytes,
        max_workers=max_workers, test=test)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    json_path = r'test.json'
    device = sys.argv[1] if len(sys.argv) > 1 else 'dml'
    preload_all(device=device)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    synthesize_audio(data, test=True, device=device)
