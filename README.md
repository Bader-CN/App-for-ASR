# Overview
App-for-ASR 是一个基于 Python 的 **实时语音识别** & **翻译** [Gradio](https://www.gradio.app/) 应用, 可以全程保持在本地运行, 不需要连接互联网


# Quickstart
- 打开方式
    - Windows 系统上双击执行 `strart_srv.bat` 即可 (需要有 `.venv` 虚拟环境)
    - 使用命令 `python app.py` 来启动服务
- 浏览器会自动打开, 如果没有请在浏览器里输入 `http://127.0.0.1:7860/` 即可


# Running effect
![ASR & Translation Running effect](./example/ASR%20and%20Translation%20effect.png)


# Requirements & Dependencies
- FFmpeg
    - 本项目需要依赖 FFmpeg, 请将 FFmpeg 的 bin 目录添加到 `config.cfg` 中的 `FFmpeg_Path`
    - FFmpeg 可以通过 [官方下载链接](https://ffmpeg.org/download.html) 来下载

- 语音识别
    - 语音识别使用的是 OpenAI Whisper 系列模型, 请将模型路径添加到 `config.cfg` 中的 `ASR_Model`
    - 模型文件需要下载到本地, 模型开源地址为:
        - HuggingFace
            - [whisper-small](https://huggingface.co/openai/whisper-small)
            - [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
            - [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
        - ModelScope (魔搭社区)
            - [whisper-small](https://www.modelscope.cn/models/openai-mirror/whisper-small)
            - [whisper-large-v3-turbo](https://www.modelscope.cn/models/openai-mirror/whisper-large-v3-turbo)
            - [whisper-large-v3](https://www.modelscope.cn/models/openai-mirror/whisper-large-v3)

- 语音翻译
    - 翻译需要连接 Ollama API 或 OpenAI API 来实现
        - 如果使用 Ollama API, 请参考 [Ollama 官网](https://ollama.com/)
        - 如果使用 OpenAI API, 并在本地运行, 请参考 [LM Studio 官网](https://lmstudio.ai/)

- 采集系统音频
    - 如果想采集系统内部声音, 需要寻找 "**立体声混音**"
    - Windows 上默认不开启该设备, 需要到系统里去开启, Win11 上的开启方法为:
      - 系统 -> 声音设置 -> 高级 -> 所有声音设备 -> 立体声混音 -> 允许


# Installation
1. 安装 [Python3](https://www.python.org/downloads/), 目前测试 Python 3.13 可用
2. [可选] 进入项目根目录, 并创建一个虚拟环境
    ```CMD
    # 在当前目录下创建一个虚拟环境
    python -m venv .venv
    # 激活虚拟环境
    cd .\.venv\Scripts && activate.bat
    ```
3. 安装依赖库
    - 自动安装
        ```CMD
        # 需要指定项目根目录的 requirements.txt
        # 最好先手动安装 PyTorch
        pip install -r <path_for_requirements.txt>
        ```
    - 手动安装
        ```CMD
        # 安装 PyTorch - 如果有支持 CUDA 的 GPU, 推荐选这个
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        # 安装 PyTorch - CPU 版本
        pip3 install torch torchvision

        # 其余依赖库
        pip install -U gradio
        pip install -U pyaudio
        pip install -U loguru
        pip install -U transformers datasets[audio] accelerate
        pip install -U silero_vad
        pip install -U opencc-python-reimplemented
        pip install -U langchain
        pip install -U langchain-ollama
        pip install -U langchain-openai
        ```