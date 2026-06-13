# -*- mode: python ; coding: utf-8 -*-
# hifiutau-engine PyInstaller spec — PyTorch 推理版
#
# 打包命令:
#   pyinstaller hifiserver_pytorch.spec
#
# 注意:
#   - 需要先将 pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/ 复制到打包目录
#   - 或打包后通过 HIFIUTAU_ENGINE_CKPT 环境变量指定 checkpoint 路径

import os
import sys

block_cipher = None

# 模型数据（根据实际路径调整）
_MODEL_DIR = r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
_HNSEP_DIR = r"hnsep_onnx"

# 收集模型数据文件
_datas = []
if os.path.isdir(_MODEL_DIR):
    _datas.append((_MODEL_DIR, _MODEL_DIR))

if os.path.isdir(_HNSEP_DIR):
    _datas.append((_HNSEP_DIR, _HNSEP_DIR))

a = Analysis(
    ['../http_syn2_pytorch.py'],
    pathex=['..'],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        # 合成管线
        'synthesis_pipeline',
        'synthesis_pipeline.fragment',
        'synthesis_pipeline.engine',
        'synthesis_pipeline.post_process',
        'synthesis_pipeline.tension_filter',
        'synthesis_pipeline.growl',
        'synthesis_pipeline.utils',
        # PyTorch splicer
        'tools',
        'tools.pytorch_splicer',
        'tools.hnsep_pytorch',
        'tools.nsf_hifigan',
        'tools.utils',
        # HN-SEP PyTorch 模型
        'hnsep',
        'hnsep.nets',
        'hnsep.layers',
        'waitress',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['build/runtime_hook.py'],
    excludes=[
        # 排除 ONNX Runtime（PyTorch 版不需要）
        'onnxruntime',
        'onnxruntime-gpu',
        'onnxruntime-directml',
        # 排除不需要的大型库
        'matplotlib',
        'pandas',
        'jupyter',
        'notebook',
        'tensorflow',
        'keras',
        'torchvision',
        # PyTorch 的 MKL 等可以进一步裁剪（可选）
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hifiserver_pytorch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='hifiserver_pytorch',
)
