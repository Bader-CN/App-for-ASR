import torch

from queue import Queue
from collections import deque
from threading import Event
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
from transformers import pipeline
from silero_vad import load_silero_vad
from langchain.chat_models import init_chat_model

from utils.log import logger
from utils.cfg import cfg

logger.info("Initializing global parameters and models...")

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
Task_ASR.clear()

# 翻译相关 - 这里必须使用引用变量(非 int/str/None), 否则其它模块导入时只是 "静态快照", 不是 "动态引用"
Task_LLM = {"TGT_LANG": None, "SYS_Prompt": None}

# ASR Tokenizer
ASR_Tokenizer = AutoTokenizer.from_pretrained(cfg.get("ASR", "ASR_Model"))
 
# ASR Pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
ASR_Model = AutoModelForSpeechSeq2Seq.from_pretrained(
    cfg.get("ASR", "ASR_Model"), 
    dtype=torch_dtype, 
    # low_cpu_mem_usage=True
)
ASR_Model.to(device)

Processor = AutoProcessor.from_pretrained(cfg.get("ASR", "ASR_Model"))
# 利用 ASR_Pipeline 来进行推理
# https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
ASR_Pipeline = pipeline(
    "automatic-speech-recognition",
    model=ASR_Model,
    tokenizer=Processor.tokenizer,
    feature_extractor=Processor.feature_extractor,
    dtype=torch_dtype,
    device=device,
)

# VAD Model
VAD_Model = load_silero_vad()
VAD_Model.to("cpu")

# LLM Model
LLM_Model = init_chat_model(
    model_provider=cfg.get("LLM", "Provider"),
    model = cfg.get("LLM", "Model_Name"),
    base_url = cfg.get("LLM", "Base_URL"),
    max_tokens = cfg.getint("LLM", "MAX_Tokens"),
    api_key = cfg.get("LLM", "API_Key"),
)
llm_init = LLM_Model.invoke("ASR translation model has been successfully loaded!")
logger.info(f"{llm_init.response_metadata}")

logger.success("Finish initialized global parameters and models.")