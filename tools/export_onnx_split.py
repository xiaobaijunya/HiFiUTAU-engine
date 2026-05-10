"""
导出 SplitGenerator 的 part1 / part2 为独立 ONNX 模型。

使用 legacy TorchScript 导出器 (dynamo=False) 以保证数值精度。

用法:
    python tools/export_onnx_split.py
        --ckpt pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt
        --output_dir exported_onnx
        --device cpu
"""

import argparse
import json
import os
import pathlib
import sys

# 确保能找到 tools 包 (nsf_hifigan 内部使用 from tools.utils import ...)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import numpy as np

from tools.nsf_hifigan import SplitGenerator, AttrDict


class Part1Wrapper(torch.nn.Module):
    """包装 forward_part1 为独立 Module"""
    def __init__(self, model: SplitGenerator):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model.forward_part1(mel)


class Part2Wrapper(torch.nn.Module):
    """包装 forward_part2 为独立 Module，保持与原始调用一致的默认参数"""
    def __init__(self, model: SplitGenerator):
        super().__init__()
        self.model = model

    def forward(self, feat, f0):
        # 与原始 synthesize() 调用一致: nsf_gain=1.0, x_gain=1.0
        return self.model.forward_part2(feat, f0, nsf_gain=1.0, x_gain=1.0, out_har=False)


# ---------------------------------------------------------------------------
# 数值验证辅助
# ---------------------------------------------------------------------------
def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    """返回两个数组的最大绝对差异"""
    return float(np.max(np.abs(a - b)))


def _relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    """返回相对差异 max(|a-b| / (|a|+1e-8))"""
    denom = np.max(np.abs(a)) + 1e-8
    return float(np.max(np.abs(a - b)) / denom)


def validate_numerical(model: SplitGenerator, wrapper: torch.nn.Module,
                       onnx_path: str, example_inputs, input_names: list[str],
                       name: str, atol: float = 1e-4, rtol: float = 1e-3):
    """比较 PyTorch 前向输出 vs ONNX Runtime 输出"""
    import onnxruntime as ort
    print(f"  ── 数值验证 {name} ──")

    # PyTorch 前向
    with torch.no_grad():
        pt_out = wrapper(*example_inputs) if isinstance(example_inputs, tuple) else wrapper(example_inputs)
    if isinstance(pt_out, (tuple, list)):
        pt_out = pt_out[0]
    pt_np = pt_out.cpu().numpy()

    # ONNX Runtime 推理
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    feeds = {}
    for i, name_i in enumerate(input_names):
        val = example_inputs[i] if isinstance(example_inputs, tuple) else example_inputs
        feeds[name_i] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
    ort_out = sess.run(None, feeds)[0]

    # 比较
    abs_diff = _max_diff(pt_np, ort_out)
    rel_diff = _relative_diff(pt_np, ort_out)
    match = abs_diff < atol and rel_diff < rtol

    print(f"    PyTorch 输出形状: {pt_np.shape}")
    print(f"    ONNX    输出形状: {ort_out.shape}")
    print(f"    最大绝对差异:    {abs_diff:.6e}")
    print(f"    相对差异:        {rel_diff:.6e}")
    if match:
        print(f"    ✅ 数值一致 (atol={atol}, rtol={rtol})")
    else:
        print(f"    ❌ 数值偏差超出阈值!")
    print()
    return match


# ---------------------------------------------------------------------------
# 导出函数
# ---------------------------------------------------------------------------
def export_part1(model: SplitGenerator, output_dir: str, opset: int = 17):
    """导出 part1: mel → 隐特征 (legacy exporter, dynamo=False)"""
    print("=" * 60)
    print("导出 part1.onnx  (mel → 隐特征)")
    print("=" * 60)

    T_mel = 100
    dummy_mel = torch.randn(1, model.h.num_mels, T_mel)
    part1_path = os.path.join(output_dir, "part1.onnx")
    device = next(model.parameters()).device
    wrapper = Part1Wrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        dummy_mel,
        part1_path,
        dynamo=False,                   # 使用 legacy TorchScript 导出器
        input_names=["mel"],
        output_names=["feat"],
        dynamic_axes={
            "mel": {2: "mel_frames"},
            "feat": {2: "feat_frames"},
        },
        opset_version=opset,
        verbose=False,
    )
    print(f"  输入: mel     (1, {model.h.num_mels}, mel_frames)")
    print(f"  输出: feat    (1, 128, feat_frames=mel_frames*64)")
    print(f"  保存到: {part1_path}")

    # 数值验证
    validate_numerical(
        model, wrapper, part1_path, dummy_mel, ["mel"],
        "part1", atol=1e-4, rtol=1e-3
    )


def export_part2(model: SplitGenerator, output_dir: str, opset: int = 17):
    """导出 part2: 隐特征 + f0 → 波形 (legacy exporter, dynamo=False)"""
    print("=" * 60)
    print("导出 part2.onnx  (隐特征 + f0 → 波形)")
    print("=" * 60)

    T_mel = 100
    T_feat = T_mel * model.upp
    dummy_feat = torch.randn(1, 128, T_feat)
    dummy_f0 = torch.randn(1, T_mel)
    part2_path = os.path.join(output_dir, "part2.onnx")
    device = next(model.parameters()).device
    wrapper = Part2Wrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (dummy_feat, dummy_f0),
        part2_path,
        dynamo=False,                   # 使用 legacy TorchScript 导出器
        input_names=["feat", "f0"],
        output_names=["waveform"],
        dynamic_axes={
            "feat": {2: "feat_frames"},
            "f0": {1: "mel_frames"},
            "waveform": {2: "audio_samples"},
        },
        opset_version=opset,
        verbose=False,
    )
    print(f"  输入: feat   (1, 128, feat_frames)")
    print(f"  输入: f0     (1, mel_frames)")
    print(f"  输出: wav    (1, 1, audio_samples=feat_frames*8)")
    print(f"  保存到: {part2_path}")

    # 数值验证
    validate_numerical(
        model, wrapper, part2_path, (dummy_feat, dummy_f0),
        ["feat", "f0"], "part2", atol=1e-4, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="导出 SplitGenerator ONNX 模型")
    parser.add_argument("--ckpt", type=str,
                        default="pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="exported_onnx",
                        help="ONNX 输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        help="导出设备 (建议 cpu 避免精度抖动)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = pathlib.Path(args.ckpt)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载配置
    config_file = ckpt_path.with_name("config.json")
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    print(f"模型配置:")
    print(f"  num_mels:        {h.num_mels}")
    print(f"  hop_size:        {h.hop_size}")
    print(f"  upsample_rates:  {h.upsample_rates}")
    print(f"  mini_nsf:        {h.mini_nsf}")
    print(f"  sampling_rate:   {h.sampling_rate}")
    print()

    # 2. 构建模型并加载权重
    generator = SplitGenerator(h)
    cp_dict = torch.load(ckpt_path, map_location="cpu")
    generator.load_state_dict(cp_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()
    generator.to(device)
    del cp_dict
    print("模型加载完成，weight norm 已移除\n")

    # 3. 导出 part1
    export_part1(generator, output_dir, args.opset)

    # 4. 导出 part2
    export_part2(generator, output_dir, args.opset)

    print(f"\n完成! ONNX 模型已保存到: {os.path.abspath(output_dir)}")
    print(f"  part1.onnx:  mel → 隐特征 (用于逐音素处理)")
    print(f"  part2.onnx:  隐特征 + f0 → 波形 (用于最终合成)")


if __name__ == "__main__":
    main()
