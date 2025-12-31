import os
import gradio as gr

from utils.gui import get_audio_devices, set_audio_dev_idx
from utils.gui import set_source_language, set_target_language
from utils.gui import set_vad_threshold, set_btn_label
from utils.cfg import cfg
from utils.log import logger
from utils.core import audio_text_update
from utils.vars import Task_VAD

# 设置环境变量, 用来将 FFmpeg 添加进入系统环境变量中
try:
    # os.pathsep 是平台相关的路径分隔符
    os.environ["Path"] = os.path.abspath(cfg.get("General", "FFmpeg_Path")) + os.pathsep + os.environ['PATH'] + os.environ["Path"]
    logger.info(f"Temporarily add Path environment variables: {os.path.abspath(cfg.get("General", "FFmpeg_Path"))}")
except Exception as e:
    logger.error(e)

# 可录音的设备列表
audio_devices = get_audio_devices()

# https://www.gradio.app/guides/controlling-layout
with gr.Blocks(title="Real-time ASR", fill_width=True) as demo:
    # equal_height=True
    with gr.Row(equal_height=True):
        # 音频输入设备
        ui_mid_driver = gr.Dropdown(choices=[k for k in audio_devices.keys()], label="输入设备", interactive=True, scale=2)
        ui_mid_driver.select(fn=set_audio_dev_idx, inputs=ui_mid_driver, outputs=None)

        # 原始语言
        ui_source_lang = gr.Dropdown(choices=["auto", "chinese", "english", "japanese"], label="原始语言", interactive=True, scale=1)
        ui_source_lang.select(fn=set_source_language, inputs=ui_source_lang, outputs=None)

        # 目标语言
        ui_target_lang = gr.Dropdown(choices=["none", "chinese", "english", "japanese"], label="目标语言", interactive=True, scale=1)
        ui_target_lang.select(fn=set_target_language, inputs=ui_target_lang, outputs=None)

        # VAD 活动语音检测的阈值
        ui_vad_threshold = gr.Slider(label="语音检测", minimum=0, maximum=1, value=Task_VAD.get("VAD_Threshold"), interactive=True, scale=1)
        ui_vad_threshold.change(fn=set_vad_threshold, inputs=ui_vad_threshold, outputs=None)

        # 开启/停止
        ui_btn_str_end = gr.Button(value="开始", interactive=True, scale=1)
        ui_btn_str_end.click(fn=set_btn_label, inputs=ui_btn_str_end, outputs=ui_btn_str_end)

    with gr.Row():
        ui_audio_content = gr.TextArea(label="识别内容", value=audio_text_update, every=cfg.getfloat("General", "ASR_Update_Freq"))

if __name__ == "__main__":
    demo.launch(
        server_port=cfg.getint("General", "Server_Port"),
        inbrowser=True,
    )