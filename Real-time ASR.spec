# -*- mode: python ; coding: utf-8 -*-

import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

# 添加 ffmpeg 到环境变量中，确保 PyInstaller 能够正确找到 FFmpeg
ffmpeg_path = r".\ffmpeg\bin"
os.environ["Path"] = os.path.abspath(ffmpeg_path) + os.pathsep + os.environ["Path"]

collect_all_list = [
    "torchcodec",
    "silero_vad",
    "langchain_ollama",
    "langchain_openai",
    # "transformers",
    # "optimum.intel",
    "openvino",
]

datas = []
binaries = []
hiddenimports = []

for packet in collect_all_list:
    packet_datas, packet_binaries, packet_hiddenimports = collect_all(packet)
    datas.extend(packet_datas)
    binaries.extend(packet_binaries)
    hiddenimports.extend(packet_hiddenimports)

datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
datas += collect_data_files('groovy')
datas += collect_data_files('safehttpx')

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
        'transformers': 'py',
        'optimum.intel': 'py',
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
