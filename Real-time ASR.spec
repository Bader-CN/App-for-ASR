# -*- mode: python ; coding: utf-8 -*-

import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

# 添加 ffmpeg 到环境变量中，确保 PyInstaller 能够正确找到 FFmpeg
ffmpeg_path = r".\ffmpeg\bin"
os.environ["Path"] = os.path.abspath(ffmpeg_path) + os.pathsep + os.environ["Path"]

datas = []
binaries = []
hiddenimports = []

datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
datas += collect_data_files('groovy')
datas += collect_data_files('safehttpx')

# 修复 torchcodec 依赖问题
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torchcodec')
datas.extend(torch_datas)
binaries.extend(torch_binaries)
hiddenimports.extend(torch_hiddenimports)

# 修复 silero_vad 依赖问题
silero_datas, silero_binaries, silero_hiddenimports = collect_all('silero_vad')
datas.extend(silero_datas)
binaries.extend(silero_binaries)
hiddenimports.extend(silero_hiddenimports)

a = Analysis(
    ['Real-time ASR.py'],
    pathex=[],
    binaries=binaries, 
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    excludes=[],
    noarchive=False,
    optimize=0,
    # gradio 依赖问题, 不能搜集 pyc 文件, 只能收集 py 文件
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Real-time ASR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Real-time ASR',
)
