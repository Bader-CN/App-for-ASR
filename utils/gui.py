import pyaudio

from utils.core import pyaudio_callback, process_asr
from utils.vars import Audio_Dev_IDX, VAD_Threshold
from utils.vars import MIC, Stream, Task_ASR
from utils.vars import ASR_Executor
from utils.vars import ASR_Audio_Buffer
from utils.vars import Generate_Kwargs
from utils.vars import Task_LLM
from utils.vars import Task_VAD
from utils.cfg import cfg
from utils.log import logger


def get_audio_devices() -> dict:
    """
    从 PyAudio 中查找所有受支持的音频设备
    """
    p = pyaudio.PyAudio()
    available_devs_dict = dict()
    for idx in range(p.get_device_count()):
        dev = p.get_device_info_by_index(idx)
        # 只显示可录音设备(hostApi 版本必须是0, 否则会报错)
        # 如果想要录制电脑声音, 需要寻找 "Microsoft 声音映射器(优先, 但可能要指定默认设备)/立体声混音"
        # 立体声混音在 Windows 上默认禁用, 需要手动开启
        if dev['maxInputChannels'] > 0 and dev['hostApi'] == 0:
            available_devs_dict[dev['name']]=dev['index']
    
    return available_devs_dict


def set_audio_dev_idx(audio_name):
    """
    设置选择的音频设备id
    """
    global Audio_Dev_IDX
    audio_devs = get_audio_devices()
    Audio_Dev_IDX = audio_devs.get(audio_name)
    logger.info(f"Selected audio dev: {audio_name}")
    logger.info(f"Selected audio idx: {Audio_Dev_IDX}")


max_tokens = cfg.getint("ASR", "ASR_Max_Tokens")

def set_source_language(language):
    """
    OpenAI Whisper API 中的 generate_kwargs 参数 - 用于指定原始语言
    """
    global Generate_Kwargs
    if language == "auto":
        Generate_Kwargs = {"max_new_tokens": max_tokens}
    else:
        Generate_Kwargs = {"max_new_tokens": max_tokens, "language":language}
    logger.info(f"generate_kwargs updated: {Generate_Kwargs}")


def set_target_language(language):
    """
    设置翻译语音的标记位和提示词
    """
    global Task_LLM
    if language == "none":
        Task_LLM["TGT_LANG"] = None
        Task_LLM["SYS_Prompt"] = cfg.get("LLM", "SYS_Prompt")
    elif language == "chinese":
        Task_LLM["TGT_LANG"] = language
        Task_LLM["SYS_Prompt"] = cfg.get("LLM", "SYS_Prompt").replace("<LANG>", "中文")
    elif language == "english":
        Task_LLM["TGT_LANG"] = language
        Task_LLM["SYS_Prompt"] = cfg.get("LLM", "SYS_Prompt").replace("<LANG>", "英文")
    elif language == "japanese":
        Task_LLM["TGT_LANG"] = language
        Task_LLM["SYS_Prompt"] = cfg.get("LLM", "SYS_Prompt").replace("<LANG>", "日语")
    logger.info(f"Currently TGT_LANG: {Task_LLM.get("TGT_LANG")}")
    logger.info(f"Currently LLM system prompt: {Task_LLM.get("SYS_Prompt")}")


def set_vad_threshold(threshold):
    """
    设置 VAD 阈值
    """
    global Task_VAD
    Task_VAD["VAD_Threshold"] = threshold
    logger.info(f"VAD Threshold updated: {Task_VAD.get("VAD_Threshold")}")


def set_btn_label(label):
    """
    启动 & 停止语音识别任务
    """
    try:
        global MIC
        global Stream
        if label == "开始":
            # 设置标记位, 开始采集音频
            Task_ASR.set()
            # 利用 PyAudio 打开音频流
            MIC = pyaudio.PyAudio()
            Stream = MIC.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                # 每个音频帧的长度
                frames_per_buffer=cfg.getint("ASR", "Frames_Per_Buffer"),
                stream_callback=pyaudio_callback,
                input=True,
                input_device_index=Audio_Dev_IDX,
            )
            Stream.start_stream()
            logger.info(f"Start Stream...")
            ASR_Executor.submit(process_asr)
            logger.info(f"Start ASR")
            return "结束"

        if label == "结束":
            # 终止 PyAudio 对象 & 音频流
            Stream.stop_stream()
            Stream.close()
            MIC.terminate()
            # 清除标记位
            Task_ASR.clear()
            # 强制清除缓存队列
            ASR_Audio_Buffer.clear()
            logger.info(f"End ASR")
            return "开始"
        
    except Exception as e:
        logger.error(e)