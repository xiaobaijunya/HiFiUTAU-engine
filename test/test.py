"""
合成测试：向已运行的 HTTP 服务推送 test.json → 验证 WAV 输出。

用法:
    python test/test.py                              # 默认 localhost:8000
    python test/test.py http://localhost:8000         # 指定服务地址
    python test/test.py --dry-run                    # 仅检查依赖
"""

import json
import os
import sys
import urllib.request
import urllib.error
import struct

SERVER_PORT = 8000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_JSON = os.path.join('test/test.json')
OUTPUT_WAV = os.path.join(BASE_DIR, 'test', 'test_out.wav')


def dry_run():
    """仅检查关键文件和依赖是否存在。"""
    missing = []
    if not os.path.isfile(TEST_JSON):
        missing.append(TEST_JSON)
    wav_dir = os.path.join(BASE_DIR, 'test', 'wav')
    if not os.path.isdir(wav_dir):
        missing.append(wav_dir)

    if missing:
        print("[DRY-RUN] 缺少文件/目录:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("[DRY-RUN] 测试环境就绪 ✓")
    sys.exit(0)


def send_request(url: str, data: dict) -> bytes:
    """POST JSON 请求并返回响应体。"""
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def validate_wav(wav_bytes: bytes) -> bool:
    """简单验证 WAV 文件头是否合法。"""
    if len(wav_bytes) < 44:
        print(f"[FAIL] WAV 文件太小: {len(wav_bytes)} bytes (< 44)")
        return False
    if wav_bytes[:4] != b'RIFF':
        print("[FAIL] 不是合法的 RIFF/WAV 文件")
        return False
    if wav_bytes[8:12] != b'WAVE':
        print("[FAIL] 缺少 WAVE 标识")
        return False
    sample_rate = struct.unpack('<I', wav_bytes[24:28])[0]
    print(f"[OK] 采样率: {sample_rate} Hz")
    data_size = struct.unpack('<I', wav_bytes[40:44])[0]
    duration = data_size / (sample_rate * 2)  # 16-bit mono
    print(f"[OK] 音频时长: {duration:.2f} 秒")
    return True


def main():
    # 服务地址
    server_url = f'http://localhost:{SERVER_PORT}'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--dry-run':
            dry_run()
        server_url = sys.argv[1]

    api_url = f'{server_url.rstrip("/")}/synthesize'

    # 检查测试数据
    if not os.path.isfile(TEST_JSON):
        print(f"[FAIL] 找不到测试数据: {TEST_JSON}")
        sys.exit(1)

    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 修改输出路径为 test 目录
    test_data['out_wav'] = OUTPUT_WAV

    # 推送合成请求
    print(f"[INFO] 发送合成请求到 {api_url} ...")
    try:
        wav_bytes = send_request(api_url, test_data)
    except urllib.error.URLError as e:
        print(f"[FAIL] 请求失败: {e}")
        print("请确保合成服务已启动，例如:")
        print("  python http_syn2_cpu.py")
        print("  python http_syn2_pytorch.py")
        sys.exit(1)

    print(f"[OK] 收到响应: {len(wav_bytes)} bytes")

    # 保存 WAV
    with open(OUTPUT_WAV, 'wb') as f:
        f.write(wav_bytes)
    print(f"[OK] WAV 已保存: {OUTPUT_WAV}")

    # 验证 WAV
    if not validate_wav(wav_bytes):
        sys.exit(1)

    print("[PASS] 合成测试通过 ✓")


if __name__ == '__main__':
    main()
