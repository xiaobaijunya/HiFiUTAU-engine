"""
导出 SplitGenerator 的 part1 / part2 为独立 ONNX 模型。

用法:
    python tools/export_onnx_split.py
        --ckpt pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt
        --output_dir exported_onnx
        --device cpu
"""
# python export_onnx_split.py --ckpt pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt --output_dir exported_onnx
import argparse
import json
import os
import pathlib

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
    """包装 forward_part2 为独立 Module"""
    def __init__(self, model: SplitGenerator):
        super().__init__()
        self.model = model

    def forward(self, feat, f0):
        return self.model.forward_part2(feat, f0)


def export_part1(model: SplitGenerator, output_dir: str, opset: int = 17):
    """导出 part1: mel → 隐特征"""
    print("=" * 60)
    print("导出 part1.onnx  (mel → 隐特征)")
    print("=" * 60)

    # 用固定长度 dummy input 做 trace
    T_mel = 100  # 100 帧 mel，可被动态轴覆盖
    dummy_mel = torch.randn(1, model.h.num_mels, T_mel)

    part1_path = os.path.join(output_dir, "part1.onnx")
    device = next(model.parameters()).device
    wrapper = Part1Wrapper(model).to(device).eval()
    torch.onnx.export(
        wrapper,
        dummy_mel,
        part1_path,
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
    print()


def export_part2(model: SplitGenerator, output_dir: str, opset: int = 17):
    """导出 part2: 隐特征 + f0 → 波形"""
    print("=" * 60)
    print("导出 part2.onnx  (隐特征 + f0 → 波形)")
    print("=" * 60)

    # 构造 dummy 输入
    # feat 时间维度 = mel_frames * 64
    T_mel = 100
    T_feat = T_mel * model.upp  # upp = 8*8 = 64
    dummy_feat = torch.randn(1, 128, T_feat)
    dummy_f0 = torch.randn(1, T_mel)

    part2_path = os.path.join(output_dir, "part2.onnx")
    device = next(model.parameters()).device
    wrapper = Part2Wrapper(model).to(device).eval()
    torch.onnx.export(
        wrapper,
        (dummy_feat, dummy_f0),
        part2_path,
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
    print()


def main():
    parser = argparse.ArgumentParser(description="导出 SplitGenerator ONNX 模型")
    parser.add_argument("--ckpt", type=str,
                        default="pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="exported_onnx",
                        help="ONNX 输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        help="导出设备 (cpu / cuda)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset 版本 (默认 18)")
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

    # 5. 验证
    print("=" * 60)
    print("验证 ONNX 模型")
    print("=" * 60)
    try:
        import onnxruntime as ort

        # 验证 part1
        onnx_path = os.path.join(output_dir, "part1.onnx")
        sess = ort.InferenceSession(onnx_path)
        print(f"  part1.onnx: 输入={[i.name for i in sess.get_inputs()]}, "
              f"输出={[o.name for o in sess.get_outputs()]}")
        test_mel_frames = 20
        feeds = {"mel": np.random.randn(1, 128, test_mel_frames).astype(np.float32)}
        out = sess.run(None, feeds)
        print(f"    推理成功! 输出形状: {out[0].shape}")

        # 验证 part2 (feat_frames = mel_frames * 64)
        onnx_path = os.path.join(output_dir, "part2.onnx")
        sess = ort.InferenceSession(onnx_path)
        print(f"  part2.onnx: 输入={[i.name for i in sess.get_inputs()]}, "
              f"输出={[o.name for o in sess.get_outputs()]}")
        test_mel_frames = 20
        test_feat_frames = test_mel_frames * generator.upp  # upp=64
        feeds = {
            "feat": np.random.randn(1, 128, test_feat_frames).astype(np.float32),
            "f0": np.random.randn(1, test_mel_frames).astype(np.float32),
        }
        out = sess.run(None, feeds)
        print(f"    推理成功! 输出形状: {out[0].shape}")
    except ImportError:
        print("  未安装 onnxruntime，跳过验证")
    except Exception as e:
        print(f"  验证出错: {e}")

    print(f"\n完成! ONNX 模型已保存到: {os.path.abspath(output_dir)}")
    print(f"  part1.onnx:  mel → 隐特征 (用于逐音素处理)")
    print(f"  part2.onnx:  隐特征 + f0 → 波形 (用于最终合成)")


if __name__ == "__main__":
    main()
