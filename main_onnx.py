"""
hifisampler ONNX 合成入口 — 模型加载 + 合成调度。

模型在此文件中加载&缓存，合成逻辑委托给 synthesis_pipeline。
"""
import json
import os

from tools.hidden_splicer import HiddenSplicer
from synthesis_pipeline import SynthesisEngine


# ============================================================================
# 全局模型缓存
# ============================================================================

_onnx_splicer: HiddenSplicer | None = None
_onnx_splicer_config: tuple | None = None
_hnsep_session = None


def get_splicer(part1_onnx: str, part2_onnx: str,
                config_path: str, device: str = 'cpu') -> HiddenSplicer:
    """获取缓存的 HiddenSplicer，首次加载后复用。"""
    global _onnx_splicer, _onnx_splicer_config
    config_key = (part1_onnx, part2_onnx, config_path, device)
    if _onnx_splicer is None or _onnx_splicer_config != config_key:
        print("[缓存] 加载 HiddenSplicer ONNX 模型...")
        _onnx_splicer = HiddenSplicer(part1_onnx, part2_onnx,
                                       config_path, device=device)
        _onnx_splicer_config = config_key
        print("[缓存] HiddenSplicer 就绪")
    return _onnx_splicer


def get_hnsep_session():
    """获取 HN-SEP 全局会话 (懒加载)。"""
    global _hnsep_session
    if _hnsep_session is None:
        from tools.hnsep_onnx import get_global_hnsep_session
        _hnsep_session = get_global_hnsep_session()
    return _hnsep_session


def preload_all(device: str = 'dml'):
    """服务器启动时预加载所有模型。

    Args:
        device: 'dml' (DirectML, 默认), 'cuda', 或 'cpu'
    """
    onnx_dir = r"exported_onnx_v2"

    # HiddenSplicer (part1/part2 适合 DML 加速)
    part1 = os.path.join(onnx_dir, "part1.onnx")
    part2 = os.path.join(onnx_dir, "part2.onnx")
    cfg = os.path.join(onnx_dir, "config.json")
    get_splicer(part1, part2, cfg, device)

    # HN-SEP (含 LSTM, DML 不兼容, 强制 CPU)
    try:
        from tools.hnsep_onnx import preload_hnsep_model
        preload_hnsep_model()
        get_hnsep_session()
    except Exception as e:
        print(f"[警告] HN-SEP 预加载失败: {e}")

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

    engine = SynthesisEngine(splicer=splicer, hnsep_session=hnsep)
    return engine.synthesize(json_data, test=test, max_workers=max_workers)


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
