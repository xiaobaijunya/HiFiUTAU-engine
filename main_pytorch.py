"""
hifiutau-engine PyTorch 合成入口 — 模型加载 + 合成调度。

与 main_onnx.py 接口完全兼容，但使用原生 PyTorch 推理。
模型在此文件中加载&缓存，合成逻辑委托给 synthesis_pipeline。
"""
import json
import os
import sys

import torch

from tools.pytorch_splicer import PytorchHiddenSplicer
from tools.hnsep_pytorch import PytorchHnsep
from synthesis_pipeline import SynthesisEngine
from synthesis_pipeline.engine import load_splicer_config
from util.wav2mel import PitchAndTimeAdjustableMelSpectrogram


# ============================================================================
# 全局模型缓存
# ============================================================================

_pytorch_splicer: PytorchHiddenSplicer | None = None
_pytorch_splicer_config: tuple | None = None
_hnsep_model: PytorchHnsep | None = None
_splicer_config = None


def get_splicer(checkpoint_path: str, config_path: str,
                device: str = 'cuda', *,
                compile_model: bool = False,
                fp16: bool = False) -> PytorchHiddenSplicer:
    """获取缓存的 PyTorch Splicer，首次加载后复用。"""
    global _pytorch_splicer, _pytorch_splicer_config
    config_key = (checkpoint_path, config_path, device, compile_model, fp16)
    if _pytorch_splicer is None or _pytorch_splicer_config != config_key:
        print("[缓存] 加载 PytorchHiddenSplicer 模型...")
        _pytorch_splicer = PytorchHiddenSplicer(
            checkpoint_path, config_path, device=device,
            compile_model=compile_model, fp16=fp16,
        )
        _pytorch_splicer_config = config_key
        print("[缓存] PytorchHiddenSplicer 就绪")
    return _pytorch_splicer


def get_hnsep_model() -> PytorchHnsep | None:
    """获取 HN-SEP 全局模型。"""
    return _hnsep_model


def get_splicer_config():
    """轻量 splicer 配置（不加载 PyTorch 模型），供 hnsep/post 阶段使用。"""
    global _splicer_config
    if _splicer_config is None:
        _splicer_config = load_splicer_config(_resolve_config_path())
    return _splicer_config


# ═══════════════════════════════════════════════════════════════
#  预加载
# ═══════════════════════════════════════════════════════════════

# PyTorch checkpoint 路径（默认路径，可通过环境变量 HIFIUTAU_ENGINE_CKPT 覆盖）
_DEFAULT_CKPT_DIR = r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
_DEFAULT_CKPT = os.path.join(_DEFAULT_CKPT_DIR, "model.ckpt")
_DEFAULT_CONFIG = os.path.join(_DEFAULT_CKPT_DIR, "config.json")

# 也兼容 exported_onnx_v2/config.json（内容相同）
_FALLBACK_CONFIG = r"exported_onnx_v2/config.json"


def _resolve_ckpt_path() -> str:
    """解析 checkpoint 路径，优先使用环境变量。"""
    env_ckpt = os.environ.get("HIFIUTAU_ENGINE_CKPT")
    if env_ckpt and os.path.isfile(env_ckpt):
        return env_ckpt
    if os.path.isfile(_DEFAULT_CKPT):
        return _DEFAULT_CKPT
    raise FileNotFoundError(
        f"找不到 PyTorch checkpoint。请设置环境变量 HIFIUTAU_ENGINE_CKPT "
        f"指向 model.ckpt 文件，或确保以下路径存在：\n"
        f"  {_DEFAULT_CKPT}"
    )


def _resolve_config_path() -> str:
    """解析 config.json 路径。"""
    if os.path.isfile(_DEFAULT_CONFIG):
        return _DEFAULT_CONFIG
    if os.path.isfile(_FALLBACK_CONFIG):
        return _FALLBACK_CONFIG
    raise FileNotFoundError(
        f"找不到 config.json。请确保以下路径之一存在：\n"
        f"  {_DEFAULT_CONFIG}\n"
        f"  {_FALLBACK_CONFIG}"
    )


def _resolve_hnsep_paths(device: str):
    """解析 HN-SEP PyTorch 模型路径。"""
    model_path = os.path.join("hnsep_onnx", "model.pt")
    config_path = os.path.join("hnsep_onnx", "config.yaml")
    # 也尝试 hnsep/vr/ 下的备用路径
    alt_model = os.path.join("hnsep", "vr", "model.pt")
    alt_config = os.path.join("hnsep", "vr", "config.yaml")

    if not os.path.isfile(model_path):
        if os.path.isfile(alt_model):
            model_path = alt_model
            config_path = alt_config
        else:
            raise FileNotFoundError(
                f"找不到 HN-SEP PyTorch 模型。请将 model.pt 放到 hnsep_onnx/ 目录。"
            )
    if not os.path.isfile(config_path):
        if os.path.isfile(alt_config):
            config_path = alt_config
        else:
            raise FileNotFoundError(
                f"找不到 HN-SEP config.yaml，请确保与 model.pt 在同一目录。"
            )
    return model_path, config_path


def preload_splicer(device: str = 'cuda', *,
                    compile_model: bool = False,
                    fp16: bool = False):
    """预加载 PytorchHiddenSplicer（主合成模型）。"""
    ckpt = _resolve_ckpt_path()
    cfg = _resolve_config_path()
    print(f"[预加载] SplitGenerator checkpoint: {ckpt}")
    print(f"[预加载] 配置文件: {cfg}")
    print(f"[预加载] 优化: compile={compile_model}, fp16={fp16}")
    get_splicer(ckpt, cfg, device, compile_model=compile_model, fp16=fp16)


def preload_hnsep(device: str = 'cuda'):
    """预加载 HN-SEP PyTorch 模型。"""
    global _hnsep_model
    try:
        hnsep_ckpt, hnsep_cfg = _resolve_hnsep_paths(device)
        print(f"[预加载] HN-SEP PyTorch 模型: {hnsep_ckpt}")
        _hnsep_model = PytorchHnsep(hnsep_ckpt, hnsep_cfg, device=device)
        print("[预加载] HN-SEP PyTorch 模型就绪")
    except Exception as e:
        print(f"[警告] HN-SEP 预加载失败（不使用后处理）: {e}")
        _hnsep_model = None


def preload_all(device: str = 'cuda', *,
               compile_model: bool = False,
               fp16: bool = False):
    """预加载全部模型（splicer + hnsep）。"""
    preload_splicer(device, compile_model=compile_model, fp16=fp16)
    preload_hnsep(device)

    # ── 显示可用设备 ──
    if torch.cuda.is_available():
        print(f"[GPU] CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] CUDA 不可用，使用 CPU")

    print("[OK] 全部模型就绪")


# ============================================================================
# 合成
# ============================================================================

def synthesize_audio(json_data: dict, *, test: bool = False,
                     max_workers: int = 4, device: str = 'cuda') -> bytes:
    """执行完整合成管线。模型从缓存获取，无需每次传入。

    Args:
        json_data:   OpenUTAU JSON 数据
        test:        是否写出测试 WAV
        max_workers: cut_audio 并行线程数
        device:      'cuda' (默认), 'cpu'

    Returns:
        WAV bytes
    """
    # 从环境变量读取优化选项
    compile_model = os.environ.get('HIFIUTAU_ENGINE_COMPILE', '0') == '1'
    fp16 = os.environ.get('HIFIUTAU_ENGINE_FP16', '0') == '1'

    splicer = get_splicer(
        _resolve_ckpt_path(),
        _resolve_config_path(),
        device,
        compile_model=compile_model,
        fp16=fp16,
    )
    hnsep = get_hnsep_model()

    mel_exc = PitchAndTimeAdjustableMelSpectrogram()
    engine = SynthesisEngine(splicer=splicer, hnsep_session=hnsep, mel_exc=mel_exc)
    return engine.synthesize(json_data, test=test, max_workers=max_workers)


# ============================================================================
# 分段合成（HiFiUTAU Local 渲染器）
# ============================================================================


def synthesize_mel(json_data: dict, *, max_workers: int = 4,
                   test: bool = False, device: str = 'cuda'):
    """分段1: mel 拼接 + 变调 + HiFi-GAN。返回 (wav_bytes, written=False)。"""
    compile_model = os.environ.get('HIFIUTAU_ENGINE_COMPILE', '0') == '1'
    fp16 = os.environ.get('HIFIUTAU_ENGINE_FP16', '0') == '1'

    splicer = get_splicer(
        _resolve_ckpt_path(), _resolve_config_path(), device,
        compile_model=compile_model, fp16=fp16,
    )
    mel_exc = PitchAndTimeAdjustableMelSpectrogram()
    engine = SynthesisEngine(splicer=splicer, hnsep_session=None, mel_exc=mel_exc)
    return engine.synthesize_mel(
        json_data, max_workers=max_workers, test=test)


def synthesize_hnsep(wav_bytes: bytes, *, device: str = 'cuda'):
    """分段2: HN-SEP 气声/谐波分离。返回 (harmonic, noise, False, False)。"""
    hnsep = get_hnsep_model()
    engine = SynthesisEngine(splicer=get_splicer_config(),
                             hnsep_session=hnsep, mel_exc=None)
    return engine.synthesize_hnsep(wav_bytes)


def synthesize_post(json_data: dict, *, wav_bytes: bytes | None = None,
                    harmonic_bytes: bytes | None = None,
                    noise_bytes: bytes | None = None,
                    max_workers: int = 4, test: bool = False,
                    device: str = 'cuda'):
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
    device = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    preload_all(device=device)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    synthesize_audio(data, test=True, device=device)
