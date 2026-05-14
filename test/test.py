"""
hei.wav 完整参数测试

使用 test_hei.json 进行合成，覆盖所有 Note_flags 和 Dynamic_parameter。
支持 ONNX (cpu/dml/cuda) 和 PyTorch 两种后端。

用法:
    python test/test_hei.py                          # ONNX CPU (默认)
    python test/test_hei.py --device cuda            # ONNX CUDA
    python test/test_hei.py --pytorch                # PyTorch CUDA
    python test/test_hei.py --pytorch --device cpu   # PyTorch CPU
    python test/test_hei.py --dry-run                # 仅验证 JSON 结构
"""

import json
import sys
import os
import argparse

# 将项目根目录加入 path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def validate_json(json_path: str) -> dict:
    """验证 JSON 结构完整性。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    required_top = ['hop_size', 'sample_rate', 'out_wav', 'phoneme_list']
    for key in required_top:
        assert key in data, f"缺少顶层字段: {key}"

    assert isinstance(data['phoneme_list'], dict), "phoneme_list 必须是 dict"
    assert len(data['phoneme_list']) > 0, "phoneme_list 不能为空"

    for pid, info in data['phoneme_list'].items():
        assert 'phoneme_name' in info, f"音素 {pid} 缺少 phoneme_name"
        assert 'phoneme_oto' in info, f"音素 {pid} 缺少 phoneme_oto"
        assert 'Note_flags' in info, f"音素 {pid} 缺少 Note_flags"
        assert 'envelope' in info, f"音素 {pid} 缺少 envelope"

        oto = info['phoneme_oto']
        for k in ('audio_file_path', 'Offset', 'Consonant', 'Cutoff', 'Preutter', 'Overlap'):
            assert k in oto, f"音素 {pid} phoneme_oto 缺少 {k}"

        # 检查音频文件存在
        wav_path = oto['audio_file_path']
        abs_wav = os.path.join(_project_root, wav_path) if not os.path.isabs(wav_path) else wav_path
        assert os.path.exists(abs_wav), f"音素 {pid} 音频文件不存在: {abs_wav}"

    print(f"[OK] JSON 验证通过: {len(data['phoneme_list'])} 个音素")
    return data


def run_onnx(json_path: str, device: str):
    """使用 ONNX 后端进行合成。"""
    from main_onnx import preload_all, synthesize_audio

    print(f"--- ONNX 后端 (device={device}) ---")
    preload_all(device=device)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    wav_bytes = synthesize_audio(data, test=True, max_workers=4, device=device)
    out_path = data.get('out_wav', 'test/test_hei_output.wav')
    if os.path.isabs(out_path):
        out_path = os.path.basename(out_path)
    out_path = os.path.join(_project_root, out_path)

    with open(out_path, 'wb') as f:
        f.write(wav_bytes)
    print(f"[OK] 合成完成: {out_path} ({len(wav_bytes)} bytes)")


def run_pytorch(json_path: str, device: str):
    """使用 PyTorch 后端进行合成。"""
    from main_pytorch import preload_all, synthesize_audio

    print(f"--- PyTorch 后端 (device={device}) ---")
    preload_all(device=device)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    wav_bytes = synthesize_audio(data, test=True, max_workers=4, device=device)
    out_path = data.get('out_wav', 'test/test_hei_output.wav')
    if os.path.isabs(out_path):
        out_path = os.path.basename(out_path)
    out_path = os.path.join(_project_root, out_path)

    with open(out_path, 'wb') as f:
        f.write(wav_bytes)
    print(f"[OK] 合成完成: {out_path} ({len(wav_bytes)} bytes)")


def print_param_summary(json_path: str):
    """打印 JSON 中的参数使用情况。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\n========== 参数使用情况 ==========")

    # 全局 Dynamic_parameter
    dp = data.get('Dynamic_parameter', {})
    print(f"\n[全局 Dynamic_parameter]")
    for key, arr in dp.items():
        vals = [v for v in set(arr) if abs(v) > 0.01] if isinstance(arr, list) else []
        print(f"  {key}: {len(arr)} 帧{' (含非零值)' if vals else ''}")

    # 每个音素的 Note_flags
    print(f"\n[各音素 Note_flags]")
    flag_keys = ['vel', 'vol', 'mod', 'shft', 'phtp', 'strt', 'splc', 'g', 'B', 'H', 'P']
    for pid, info in data['phoneme_list'].items():
        nf = info.get('Note_flags', {})
        used = {k: nf.get(k) for k in flag_keys if k in nf and nf.get(k) != 0}
        print(f"  音素 {pid} ({info['phoneme_name']}): {used}")

    # 各音素的 phoneme_oto
    print(f"\n[各音素 OTO 参数]")
    for pid, info in data['phoneme_list'].items():
        oto = info.get('phoneme_oto', {})
        print(f"  音素 {pid} ({info['phoneme_name']}): "
              f"Offset={oto.get('Offset')}ms, Consonant={oto.get('Consonant')}ms, "
              f"Cutoff={oto.get('Cutoff')}ms, Preutter={oto.get('Preutter')}ms, "
              f"Overlap={oto.get('Overlap')}ms")

    print("\n================================")


def main():
    parser = argparse.ArgumentParser(description='hei.wav 完整参数测试')
    parser.add_argument('--json', default='test/test_hei.json',
                        help='测试 JSON 路径')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'cuda', 'dml'],
                        help='推理设备')
    parser.add_argument('--pytorch', action='store_true',
                        help='使用 PyTorch 后端 (默认 ONNX)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅验证 JSON 结构，不合成')
    args = parser.parse_args()

    json_path = os.path.join(_project_root, args.json)
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON 文件不存在: {json_path}")
        sys.exit(1)

    # 验证 JSON
    data = validate_json(json_path)
    print_param_summary(json_path)

    if args.dry_run:
        print("\n[Dry-Run] JSON 验证通过，跳过合成")
        return

    # 检查音频文件，缺失则自动生成
    for pid, info in data['phoneme_list'].items():
        wav_path = info['phoneme_oto']['audio_file_path']
        abs_wav = os.path.join(_project_root, wav_path) if not os.path.isabs(wav_path) else wav_path
        if not os.path.exists(abs_wav):
            print(f"[INFO] 音频文件不存在，自动生成: {abs_wav}")
            import soundfile as sf
            import numpy as np
            sr = 44100
            dur = 0.8
            t = np.linspace(0, dur, int(sr*dur), endpoint=False)
            noise = np.random.randn(len(t)) * 0.1
            noise[:int(0.1*sr)] *= 2.0
            f0 = np.linspace(200, 300, len(t))
            sine = 0.5*np.sin(2*np.pi*f0*t) + 0.25*np.sin(2*np.pi*f0*2*t) + 0.125*np.sin(2*np.pi*f0*3*t)
            wav = sine + noise
            wav[:int(0.005*sr)] *= np.linspace(0, 1, int(0.005*sr))
            wav[-int(0.01*sr):] *= np.linspace(1, 0, int(0.01*sr))
            sf.write(abs_wav, wav.astype(np.float32), sr)
            print(f"[OK] 已生成: {abs_wav} ({dur*1000:.0f}ms)")

    # 执行合成
    try:
        if args.pytorch:
            run_pytorch(json_path, args.device)
        else:
            run_onnx(json_path, args.device)
    except Exception as e:
        print(f"[ERROR] 合成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[Done] 测试完成")


if __name__ == '__main__':
    main()
