"""
合成基准测试：测试 torch.compile 启用/禁用时的合成速度。

用法:
    python test/benchmark.py                    # 默认 compile=0, 5次
    python test/benchmark.py --compile 1        # compile=1
    python test/benchmark.py --count 10         # 合成次数
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVER_PORT = 8000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_JSON = os.path.join('test/test.json')
OUTPUT_WAV = os.path.join(BASE_DIR, 'test', 'bench_out.wav')

RUNS = 5
COMPILE = 0


def run_synthesis(url: str, data: dict) -> float:
    """执行一次合成，返回耗时（秒）。"""
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        wav_bytes = resp.read()
    elapsed = time.perf_counter() - start

    # 保存 WAV
    with open(OUTPUT_WAV, 'wb') as f:
        f.write(wav_bytes)

    return elapsed, len(wav_bytes)


def main():
    global RUNS, COMPILE

    # 解析参数
    args = iter(sys.argv[1:])
    for arg in args:
        if arg == '--compile':
            COMPILE = int(next(args))
        elif arg == '--count':
            RUNS = int(next(args))
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)

    # 加载测试数据
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_data['out_wav'] = OUTPUT_WAV

    # 测试
    api_url = f'http://localhost:{SERVER_PORT}/synthesize'
    compile_label = f'compile={"ON" if COMPILE else "OFF"}'

    print(f"{'='*60}")
    print(f"  基准测试: compile={'1' if COMPILE else '0'}  |  次数: {RUNS}")
    print(f"{'='*60}")

    times = []
    sizes = []
    for i in range(RUNS):
        elapsed, size = run_synthesis(api_url, test_data)
        times.append(elapsed)
        sizes.append(size)
        print(f"  第 {i+1:2d} 次: {elapsed:.3f} 秒  |  {size/1024:.1f} KB")

    # 统计
    avg = sum(times) / len(times)
    best = min(times)
    worst = max(times)
    total_data = sum(sizes) / 1024 / 1024

    print(f"{'-'*60}")
    print(f"  平均: {avg:.3f} 秒")
    print(f"  最快: {best:.3f} 秒")
    print(f"  最慢: {worst:.3f} 秒")
    print(f"  总数据: {total_data:.1f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
