import os
import time
import torch

from queue import Queue
from collections import deque
from threading import Event
from concurrent.futures import ThreadPoolExecutor

from silero_vad import load_silero_vad

from utils.log import logger
from utils.cfg import cfg

# 仅作笔记记录, 这部分包已经按照不同情况在各自子线程里加载了
# 需求是懒加载, 未来等 PEP 810 – Explicit lazy imports 实现后在调整
# https://peps.python.org/pep-0810/
# from optimum.intel import OVModelForSpeechSeq2Seq
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
# from transformers import pipeline

logger.info("Initialize other parameters...")
#############################################
# 队列相关
Audio_Queue = Queue()                                                           # 语音切片队列
ASR_Result_Queue = Queue()                                                      # ASR 识别结果队列
ASR_Audio_Buffer = deque(maxlen=cfg.getint("ASR", "ASR_Frames_Buffer"))         # ASR 识别缓存队列

# 初始化需要的设定值
ASR_QCheck_Freq = cfg.getfloat("General", "ASR_QCheck_Freq")                    # ASR 队列检查间隔时间
ASR_Result = ""                                                                 # ASR 识别文本
ASR_Result_History = deque(maxlen=cfg.getint("General", "ASR_Result_History"))  # ASR 历史信息保留的数量
Generate_Kwargs = {"max_new_tokens": cfg.getint("ASR", "ASR_Max_Tokens")}       # ASR 任务参数
MIC = None                                                                      # PyAudio 对象
Stream = None                                                                   # 音频数据流
Audio_Dev_IDX = None                                                            # 音频设备索引号
VAD_Threshold = cfg.getfloat("ASR", "VAD_Threshold")                            # VAD 阈值
Task_VAD = {"VAD_Threshold": cfg.getfloat("ASR", "VAD_Threshold")}              # VAD 阈值

# 线程池
ASR_Executor = ThreadPoolExecutor(max_workers=1)                                # 语言识别线程池
LLM_Executor = ThreadPoolExecutor(max_workers=1)                                # 翻译线程池

# 事件标记
Task_ASR = Event()                                                              # 设置则意味着开始语言识别, 反之则取消语言识别

# 翻译相关 - 这里必须使用引用变量(非 int/str/None), 否则其它模块导入时只是 "静态快照", 不是 "动态引用"
Task_LLM = {"TGT_LANG": None, "SYS_Prompt": None}
#############################################
logger.success("Finish initialized other parameters.")


def init_ASR():
    """
    初始化 ASR 的 tokenizer, model, pipeline
    """
    logger.info("Initializing ASR tokenizer, model and pipeline...")

    # 尝试以 OpenVINO 方式加载模型
    if cfg.getboolean("INTEL", "OpenVINO_Enable"):
        logger.info("OpenVINO is enable, Load the ASR model using Optimum Intel.")
        # 加载特定包
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor, AutoTokenizer
        from transformers import pipeline
        # ASR Tokenizer
        ASR_Tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(cfg.get("INTEL", "OpenVINO_ASR_Model")))
        # Processor
        Processor = AutoProcessor.from_pretrained(os.path.abspath(cfg.get("INTEL", "OpenVINO_ASR_Model")))
        # ASR Model
        ASR_Model = OVModelForSpeechSeq2Seq.from_pretrained(os.path.abspath(cfg.get("INTEL", "OpenVINO_ASR_Model")))
        ASR_Model.to(cfg.get("INTEL", "OpenVINO_Device"))
        # ASR_Pipeline
        ASR_Pipeline = pipeline(
            "automatic-speech-recognition",
            model=ASR_Model,
            tokenizer=Processor.tokenizer,
            feature_extractor=Processor.feature_extractor,
            device="cpu",   # pipeline 强制指定 cpu, 经测试这样 Intel 的 GPU 运行才正常
        )
        logger.success("Finish initialized ASR tokenizer, model and pipeline.")

    # 正常方式加载模型
    else:
        logger.info("OpenVINO is disable, Load the ASR model using the normal method.")
        # 加载特定包
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
        from transformers import pipeline
        # ASR Tokenizer
        ASR_Tokenizer = AutoTokenizer.from_pretrained(cfg.get("ASR", "ASR_Model"))
        # Processor
        Processor = AutoProcessor.from_pretrained(cfg.get("ASR", "ASR_Model"))
        # ASR Model
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        ASR_Model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.get("ASR", "ASR_Model"), 
            dtype=torch_dtype, 
        )
        ASR_Model.to(device)
        # ASR_Pipeline
        ASR_Pipeline = pipeline(
            "automatic-speech-recognition",
            model=ASR_Model,
            tokenizer=Processor.tokenizer,
            feature_extractor=Processor.feature_extractor,
            dtype=torch_dtype,
            device=device,
        )
        logger.success("Finish initialized ASR tokenizer, model and pipeline.")

    return ASR_Tokenizer, ASR_Model, ASR_Pipeline
    

# ASR_Tokenizer, ASR_Model, ASR_Pipeline = ASR_Executor.submit(init_ASR).result()
ASR_Init = ASR_Executor.submit(init_ASR)


# LLM Model
def init_LLM():
    """
    初始化 LLM
    """
    logger.info("Initializing LLM...")
    # 导包也放进线程池里
    from langchain.chat_models import init_chat_model
    LLM_Model = init_chat_model(
        model_provider=cfg.get("LLM", "Provider"),
        model = cfg.get("LLM", "Model_Name"),
        base_url = cfg.get("LLM", "Base_URL"),
        max_tokens = cfg.getint("LLM", "MAX_Tokens"),
        api_key = cfg.get("LLM", "API_Key"),
    )
    llm_init = LLM_Model.invoke("ASR translation model has been successfully loaded!")
    logger.success(f"Finish initialized LLM: {llm_init.response_metadata}")
    return LLM_Model

# LLM_Model = LLM_Executor.submit(init_LLM).result()
LLM_Init = LLM_Executor.submit(init_LLM)

# VAD Model
VAD_Model = load_silero_vad()
VAD_Model.to("cpu")

# 异步初始化结果
init_llm = True
init_asr = True
while init_llm is True or init_asr is True:
    if LLM_Init.done():
        LLM_Model = LLM_Init.result()
        init_llm = False
    
    if ASR_Init.done():
        ASR_Tokenizer, ASR_Model, ASR_Pipeline = ASR_Init.result()
        init_asr = False
    
    time.sleep(0.2)