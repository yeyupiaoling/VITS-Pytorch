import argparse
import os
import time

import soundfile

from mvits.predict import MVITSPredictor
from mvits.utils.utils import print_arguments

parser = argparse.ArgumentParser(description='vits inference')
# 必须参数
parser.add_argument('-c', '--config_path', type=str, default="configs/config.yml", help='配置文件路径')
parser.add_argument('-m', '--model_path', type=str, default="models/latest/g_net.pth", help='模型路径')
parser.add_argument('-o', '--output_path', type=str, default="output/", help='输出文件路径')
parser.add_argument('-l', '--language', type=str, default="普通话", help='输入的语言')
parser.add_argument('-t', '--text', type=str, default='你好，我是智能语音助手', help='输入文本')
parser.add_argument('-s', '--spk', type=str, default='标准女声', help='合成目标说话人名称')
# 可选参数
parser.add_argument('-ns', '--noise_scale', type=float, default=.667, help='感情变化程度')
parser.add_argument('-nsw', '--noise_scale_w', type=float, default=0.6, help='音素发音长度')
parser.add_argument('-ls', '--speed', type=float, default=1, help='语速')
args = parser.parse_args()
print_arguments(args=args)

predictor = MVITSPredictor(configs=args.config_path, model_path=args.model_path)


def main():
    # 生成音频
    audio, sampling_rate = predictor.generate(text=args.text, spk=args.spk, language=args.language,
                                              noise_scale=args.noise_scale, noise_scale_w=args.noise_scale_w,
                                              speed=args.speed)
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, f'{int(time.time())}_{args.language}_{args.spk}.wav')
    soundfile.write(save_path, audio, sampling_rate)
    print(f'音频保存在：{save_path}')


if __name__ == "__main__":
    main()
