# -*- mode: python ; coding: utf-8 -*-
# hifiutau-engine PyInstaller spec — Linux CPU 版
# 由 GitHub Actions CI (ubuntu-latest) 自动构建

import os

block_cipher = None

a = Analysis(
    ['../http_syn2_cpu.py'],
    pathex=['..'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numba',
        'synthesis_pipeline',
        'synthesis_pipeline.fragment',
        'synthesis_pipeline.engine',
        'synthesis_pipeline.post_process',
        'synthesis_pipeline.tension_filter',
        'synthesis_pipeline.growl',
        'synthesis_pipeline.utils',
        'tools',
        'tools.hidden_splicer',
        'tools.hnsep_onnx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['build/runtime_hook.py'],
    excludes=[
        'torch',
        'torchvision',
        'matplotlib',
        'pandas',
        'jupyter',
        'notebook',
        'tensorflow',
        'keras',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hifiserver_linux',
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
    name='hifiserver_linux',
)
