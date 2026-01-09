import os
import argparse

from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, AutoTokenizer

def convert_model(src, dst):
    """
    将 PyTorch 的模型转换为 OpenVINO 支持的格式
    :param src: 源模型路径
    :param dst: 目标模型路径, 如果不填写会自动在 src 上追加 _openvino
    :return:
    """
    # 处理路径
    try:
        src = os.path.abspath(src)
        if dst == "":
            dst = src + "_openvino"
        else:
            dst = os.path.abspath(dst)

        # ASR Tokenizer
        ASR_Tokenizer = AutoTokenizer.from_pretrained(src)
        # Processor
        Processor = AutoProcessor.from_pretrained(src)
        # ASR Model
        ASR_Model = OVModelForSpeechSeq2Seq.from_pretrained(src, load_in_8bit=True, export=True)

        ASR_Tokenizer.save_pretrained(dst)
        Processor.save_pretrained(dst)
        ASR_Model.save_pretrained(dst)
        print("Model converted successfully!")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper model to OpenVINO format")
    parser.add_argument("--src", type=str, required=True, help="Source model path")
    parser.add_argument("--dst", type=str, default="", help="Converted model path")

    args = parser.parse_args()

    convert_model(args.src, args.dst)