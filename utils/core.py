import copy
import time
import difflib

import torch
import pyaudio
import numpy as np

from datetime import datetime
from collections import deque
from opencc import OpenCC
from langchain.messages import HumanMessage, SystemMessage

from utils.cfg import cfg
from utils.log import logger
from utils.vars import VAD_Model
from utils.vars import LLM_Model
from utils.vars import ASR_Pipeline
from utils.vars import LLM_Executor
from utils.vars import ASR_QCheck_Freq
from utils.vars import ASR_Result, ASR_Result_History
from utils.vars import ASR_Result_Queue, ASR_Audio_Buffer, Audio_Queue
from utils.vars import Task_ASR, Generate_Kwargs
from utils.vars import Task_LLM
from utils.vars import Task_VAD

def pyaudio_callback(in_data, frame_count, time_info, status):
    """
    PyAudio 回调函数, 用于处理音频数据
    """
    logger.debug(f"VAD_Threshold: {Task_VAD.get("VAD_Threshold")}")
    try:        
        # step1 - 对音频数据进行处理
        np_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float16) / 32768.0
        audio_data = {"input_data": np_data, "time_info": datetime.now().strftime("%H:%M:%S")}

        # step2 - 判断是否包含活动语音
        vad_data = torch.from_numpy(np_data).to(dtype=torch.float16).reshape(-1, 512).to("cpu")
        with torch.no_grad():
            max_vad_threshold = 0
            for i in range(vad_data.shape[0]):
                # 16000 是采样率
                speech_prob = VAD_Model(vad_data[i,:], 16000).item() 
                if speech_prob > max_vad_threshold:
                    max_vad_threshold = speech_prob
            # 如果没有人声
            if max_vad_threshold < Task_VAD.get("VAD_Threshold"):
                logger.trace(f"No voice detected, current value: {max_vad_threshold:.4f}")
                # 没人声时清除音频缓存队列
                if len(ASR_Audio_Buffer) > 0:
                    ASR_Audio_Buffer.clear()
                return (None, pyaudio.paContinue)

        # step3 - 如果有人声, 则将数据放入队列中
        logger.trace(f"Found voice detected, current value: {max_vad_threshold:.4f}")
        Audio_Queue.put(audio_data)
        return (None, pyaudio.paContinue)                 

    except Exception as e:
        logger.error(e)
        return (None, pyaudio.paAbort)


cc = OpenCC('t2s')                                      # 维持翻译文本, 将繁体中文转换为简体中文
delay_buffer = cfg.getint("ASR", "ASR_Delay_Buffer")    # 首次识别的延迟

def process_asr():
    """
    处理音频数据, 并送入模型进行识别
    """
    try:
        # 如果设置了标记位, 则代表录音已经开始
        while Task_ASR.is_set():
            # 队列为空, 跳过本次循环
            if Audio_Queue.empty():
                logger.debug(f"Audio_Queue is empty, skip loop...")
                time.sleep(ASR_QCheck_Freq)
                continue
            # 队列非空 & 处理数据
            else:
                # 提取原始数据
                audio_data = Audio_Queue.get()
                Audio_Queue.task_done()     # 提示数据已经处理完毕
                input_data = audio_data["input_data"]
                time_info = audio_data["time_info"]

                # 将音频数据放入缓存队列中
                ASR_Audio_Buffer.append(input_data)
                logger.debug(f"Currently ASR_Audio_Buffer Size: {len(ASR_Audio_Buffer)}")

                # 延迟判断 & 语音识别
                ASR_Audio_Buffer_Size = len(ASR_Audio_Buffer)
                if ASR_Audio_Buffer_Size >= delay_buffer:
                    # 出现延迟
                    if Audio_Queue.qsize() > 1:
                        logger.debug("Found Delay in Audio Buffer Queue, will quickly update...")
                        for i in range(Audio_Queue.qsize()):
                            audio_data = Audio_Queue.get()
                            time_info = audio_data["time_info"] # 需要修正时间
                            ASR_Audio_Buffer.append(audio_data["input_data"])
                
                    # 语音识别
                    audio_slice = np.concatenate(copy.deepcopy(ASR_Audio_Buffer), axis=0)
                    asr_src_result = ASR_Pipeline(inputs=audio_slice, generate_kwargs=Generate_Kwargs)
                    # 繁简转换, 解决异常中文问题
                    asr_src_result["text"] = cc.convert(asr_src_result.get("text"))
                    # 数据 & 时间 & 语音块长度
                    asr_src_result["time"] = time_info
                    asr_src_result["size"] = ASR_Audio_Buffer_Size
                    ASR_Result_Queue.put(asr_src_result)

    except Exception as e:
        logger.error(e)


def full_asr_text(ASR_Result, ASR_Result_History):
    """
    结合历史信息和当前信息来填充文本
    """
    history = ""
    for his_dict in ASR_Result_History:
        # 原始内容
        history += his_dict.get("asr_text") + "\n"
        # 翻译内容
        if his_dict.get("tran_text") != None:
            history += his_dict.get("tran_text") + "\n"
    
    return history + ASR_Result


def tran_asr_history():
    """
    翻译最新的历史信息
    """
    global Task_LLM
    
    if Task_LLM.get("TGT_LANG") is not None:
        # 待翻译内容
        logger.debug(f"Start translation, Task_LLM: {Task_LLM}")
        asr_text = ASR_Result_History[-1].get("asr_text")
        
        if len(ASR_Result_History) > 0:
            messages = [
                SystemMessage(content=Task_LLM.get("SYS_Prompt")),
                HumanMessage(content=asr_text)
            ]
            respone = LLM_Model.invoke(messages)
            tran_text = respone.content
            logger.trace(f"Translation content: {tran_text}")
        
        # 修改结果
        for his_dict in ASR_Result_History:
            if his_dict.get("asr_text") == asr_text:
                his_dict["tran_text"] = tran_text
                logger.debug(f"Translation complete!")


# 针对 audio_text_update 的变量
asr_size = cfg.getint("ASR", "ASR_Frames_Buffer")   # 这里用于统计音频长度, 基于此计算是否无损 & 新句子
new_time = None                                     # 新句子的开始时间
best_baseline = None                                # 最准确的基本内容                                   
candidate_list = deque(maxlen=asr_size)             # 候选列表, 用于判断最初的预测是否合理
candidate_size = 0                                  # 候选列表长度计数
log_level = cfg.get("General", "LogLevel")          # 日志等级

def audio_text_update():
    """
    将识别出来的内容更新到 GUI 上
    """
    global ASR_Result
    global ASR_Result_History
    global ASR_Audio_Buffer
    global new_time
    global best_baseline
    global candidate_list
    global candidate_size

    # 没有数据 - 队列为空
    if ASR_Result_Queue.empty():
        full_text = full_asr_text(ASR_Result, ASR_Result_History)
        return full_text
    
    # 判断延迟 & 清理延迟数据
    ASR_Result_Queue_Size = ASR_Result_Queue.qsize()
    if ASR_Result_Queue_Size > 1:
        logger.warning("Found Delay in Audio Result Queue, will quickly update...")
        # 连续提取至倒数第二个来规避延迟
        for _ in range(ASR_Result_Queue_Size - 1):
            _ = ASR_Result_Queue.get()
            ASR_Result_Queue.task_done()

    # 获取数据 - 仅针新数据 (最后一个)
    asr_src_result = ASR_Result_Queue.get()
    asr_src_result["text"] = asr_src_result.get("text").lstrip()
    ASR_Result_Queue.task_done()
    
    # 情况1: 无损情况 - 只添加 candidate_list, 不做判断
    if asr_src_result.get("size") < asr_size:
        # 计算当前候选列表的长度
        current_candidate_size = len(candidate_list)
        # 列出所有候选列表
        if log_level == "TRACE":
            logger.trace(f"Show candidate list:")
            for i in candidate_list:
                logger.trace(f"Candidate list: {i}")

        # 情况1-1: 新句子的开始; asr_size < 队列数量, 并且 len(candidate_list) = 0 时
        # 由于此时时间是上一个时刻的, 因此也追加历史记录
        if current_candidate_size == 0:
            # 保留历史记录
            if ASR_Result != "":
                logger.debug(f"Triggering case 1-1")
                logger.trace(f"New History: {ASR_Result}")
                ASR_Result_History.append({'asr_text': ASR_Result})
                ASR_Result = ""
                candidate_size = 0
                candidate_list.clear()
            # 记录新的时间
            new_time = asr_src_result.get("time")
            # 启动翻译任务
            LLM_Executor.submit(tran_asr_history)

        # 情况1-2: 新句子的开始; 上一个句子不长, 只有 candidate_list 但没有选出 best_baseline
        elif current_candidate_size < candidate_size and ASR_Result != "":
            # 保留历史记录
            logger.debug(f"Triggering case 1-2")
            logger.trace(f"New History: {ASR_Result}")
            ASR_Result_History.append({'asr_text': ASR_Result})
            candidate_size = 0
            candidate_list.clear()
            ASR_Result = ""
            # 记录新时间
            new_time = asr_src_result.get("time")
            # 启动翻译任务
            LLM_Executor.submit(tran_asr_history)
        
        # 情况1-3: 通过日志发现可能会存在 Candidate list 包含 2 段信息, 如果存在则直接去掉一段
        max_cand_size = 0
        split_idx = None
        candidate_list_copy = None
        for idx, candidate in enumerate(candidate_list):
            size = candidate.get("size")
            if size > max_cand_size:
                max_cand_size = size
            else:
                ASR_History = f"{candidate_list[idx-1].get("time")} - {candidate_list[idx-1].get("text").lstrip()}"
                ASR_Result_History.append({'asr_text': ASR_History})
                logger.debug(f"Triggering case 1-3")
                logger.trace(f"New History: {ASR_History}")
                # 启动翻译任务
                LLM_Executor.submit(tran_asr_history)
                # 准备处理历史数据
                candidate_list_copy = copy.deepcopy(candidate_list)
                split_idx = idx
                candidate_list.clear()
                break
        if candidate_list_copy is not None:
            for i in range(split_idx, len(candidate_list_copy)):
                candidate_list.append(candidate_list_copy[i])
            # 刷新时间
            new_time = asr_src_result.get("time")
        # 重置变量
        current_candidate_size = len(candidate_list)
        candidate_size = len(candidate_list)
        candidate_list_copy = None
        
        # 更新候选列表
        candidate_list.append(asr_src_result)
        candidate_size += 1
        # 抹除 best_baseline, 因为是新句子
        best_baseline = None
        # 当前完整的内容
        ASR_Result = f"{new_time} - {asr_src_result.get("text")}"
        logger.trace(f"ASR new sentence: {ASR_Result}")
        # 追加历史信息
        full_text = full_asr_text(ASR_Result, ASR_Result_History)
        return full_text
    
    # 情况2: 有损情况 - 更新 candidate_list, 并基于 candidate_list 推算出最优的 best_baseline
    else:
        logger.debug(f"Currently asr_src_result size: {asr_src_result.get('size')}")
        # candidate_list=1: 说明有延迟, 则 best_baseline 只能以这条为准
        if len(candidate_list) == 1:
            best_baseline = candidate_list[0].get("text")
            logger.trace(f"Candidate_list=1; Best BaseLine: {best_baseline}")
        # candidate_list>1: 说明正常, 存在多条完整记录, 则 best_baseline 需要进行筛选
        elif len(candidate_list) > 1:
            same_start_count = 0
            # 反向查找: 先从最长的记录开始
            # list 用来复制一份, 防止出现 RuntimeError: deque mutated during iteration 错误
            for history in list(reversed(candidate_list)):
                for temp_history in candidate_list:
                    # 比较 history 和 temp_history 的相似度
                    diff_result = difflib.SequenceMatcher(None, history.get("text"), temp_history.get("text")).get_matching_blocks()
                    # 如果相似则 长度(size!=0); 并且都有相同的开始(a=0 and b=0)
                    if diff_result[0].a == 0 and diff_result[0].b == 0 and diff_result[0].size >= 1:
                        same_start_count+=1
                # 情况2-1: 能找到1个以上, 证明非自己, 则用匹配的 history 作为 best_baseline
                if same_start_count >=1:
                    best_baseline = history.get("text")
                    logger.trace(f"Candidate_list={len(candidate_list)}; Best BaseLine: {best_baseline}")
                    break
                # 情况2-2: 找不到1个以上, 弃用 candidate_list 中的数据, 直接以最新的数据作为 best_baseline
                else:
                    best_baseline = asr_src_result.get("text")
                    logger.debug(f"Unable to select the Best Base Line, will roll back to the last one.")
        # candidate_list=0: best_baseline 已经选出, 直接跳过
        else:
            logger.debug(f"No need to select Best BaseLine.")
        # 抹除候选列表, 因为已经筛选出对应的 best_baseline
        candidate_list.clear()
    
    # 情况3: best_baseline 不为 None, 则说明是一个长的历史语句, 则更新 best_baseline 到最新识别的内容
    if best_baseline is not None:
        asr_now_result = asr_src_result.get("text")
        diff_result = difflib.SequenceMatcher(None, best_baseline, asr_now_result).get_matching_blocks()
        
        # 情况3-1: 有相似的内容
        if len(diff_result) > 1 and len(best_baseline) != len(asr_now_result):
            s = diff_result[-2].b
            e = diff_result[-2].size
            new_content = asr_now_result[s+e:]          # 新识别的内容
            best_baseline = best_baseline + new_content # 完整的最新记录
            logger.trace(f"Best BaseLine update: {best_baseline}")
        # 情况3-2: 没有相似的内容, 不动 best_baseline 的值
        else:
            logger.debug(f"ASR Restult will not update, Because no new content detected.")

    # 追加历史信息 & 返回结果
    ASR_Result = f"{new_time} - {best_baseline}"
    logger.trace(f"ASR current sentence: {ASR_Result}")
    full_text = full_asr_text(ASR_Result, ASR_Result_History)
    return full_text