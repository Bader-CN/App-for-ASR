# Overview
App-for-ASR 是一个基于 Python 的 **实时语音识别** & **翻译** [Gradio](https://www.gradio.app/) 应用
- 可以全程保持在本地运行, 不需要连接互联网
- 支持 Intel GPU, 可以将 OpenAI Whisper 系列模型转换成 OpenVINO 支持的格式
  - 如果是 Intel 核显, 建议选择 `whisper-tiny` 和 `whisper-base` 来进行转换
  - 使用前需要手动转换模型, 命令如下:
    ```CMD
    # 激活虚拟环境
    cd .\.venv\Scripts && activate.bat
    # 转换模型
    python scripts\convert_model.py --src <model_path> [--dst <convert_model>]
    ```

# Quickstart
- 前置准备
  - 编辑配置文件 `config.cfg`, 需要指定 `FFmpeg_Path` 对应的路径 (需要 Full-Shared 版本)
  - 编辑配置文件 `config.cfg`, 需要指定 `ASR_Model` 模型对应的路径
  - 编辑配置文件 `config.cfg`, 需要指定 `LLM` 部分对应的参数
  - 如果想使用 Intel 显卡来进行 ASR, 可以在 `config.cfg` 中设置 `OpenVINO_Enable=True`, 并指定 `OpenVINO_ASR_Model` 为转换后的模型路径
- 打开方法
  - 执行 `Real-time ASR.bat`        (请参考 Installation 来配置环境)
  - 执行 `python Real-time ASR.py`  (请参考 Installation 来配置环境)
  - 执行编译后的 `Real-time ASR.exe` (请参考 How to Build 部分)

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
    - 如果想采集系统内部声音, 需要寻找 "**立体声混音 / Stereo Mix**"
    - Windows 上默认不开启该设备, 需要到系统里去开启, Win11 上的开启方法为:
      - 系统 -> 声音设置 -> 高级 -> 所有声音设备 -> 立体声混音 -> 允许
    - 如果存在外部麦克风/耳机, 可能会导致 "**立体声混音 / Stereo Mix**" 失效, 如果外部麦克风/耳机必须要使用, 则可以尝试 [Voicemeeter](https://voicemeeter.com/) 进行音频路由
      - 详细介绍请参考 [VOICEMEETER Virtual Inputs/Outputs (VAIO)](https://voicemeeter.com/quick-tips-voicemeeter-virtual-inputs-and-outputs-windows-10-and-up/)
      - 如果只想采集电脑音频的话, Standard 版本就足够了
      - 安装完成后, 在 Windows 系统中将 "声音输出" 设置为 `Voicemeeter Input (VB-Audio Voicemeeter VAIO)`
      - 设置完成后启动 Voicemeeter, 将最右边的 A1 设置为目前正在发声的设备 (最右侧的 Hardware Out 部分)
      - 此时在软件界面里, B(Virtual Out) 就会出现声音, 使用软件采集 B1 设备即可采集电脑音频
        - A 系列设备是物理设备 (输出设备), Standard 版本可以同时选择2个, 这里只用 A1 即可
        - B 系列设备是虚拟设备 (输入设备), Standard 版本只有 B1
      - Voicemeeter Standard 版本只启用了2个A设备和1个B设备, 如果只打算采集电脑音频, 那么可以只启用这两个设备, 其余都可以禁用
        - 输出设备
          - `Voicemeeter Input (VB-Audio Voicemeeter VAIO)`
        - 输入设备
          - `Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)`


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
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        # 安装 PyTorch - CPU 版本
        pip install torch torchvision

        # 其余依赖库
        # 如果安装缓慢, 可以指定镜像站: -i https://pypi.tuna.tsinghua.edu.cn/simple
        pip install -U gradio
        pip install -U pyaudio
        pip install -U loguru
        pip install -U transformers datasets[audio] accelerate
        pip install -U silero_vad
        pip install -U opencc-python-reimplemented
        pip install -U langchain
        pip install -U langchain-ollama
        pip install -U langchain-openai
        pip install -U pywebview
        pip install -U pyinstaller
        pip install -U --upgrade-strategy eager optimum[openvino]
        ```


# How to Build
> 请确保已经安装好了环境依赖部分
1. 激活虚拟环境
  ```CMD
  cd .\.venv\Scripts && activate.bat
  ```
2. 编辑 `Real-time ASR.spec` 文件中的第7行 `ffmpeg_path`，修改为你的 FFmpeg Full-Shared 路径
3. 启动构建命令, 完成后会在 `dist` 目录下生成可执行文件
  ```
  pyinstaller ".\Real-Time ASR.spec"
  ```
4. 将文件 `config.json` 和目录 `model` / `FFmpeg` 复制到 `dist\Real-time ASR` 目录下
5. 调整  `config.json` 相关项, 确保能够找到 `模型目录 model`, `FFmpeg` 的路径, 以及 `LLM` 部分参数正确