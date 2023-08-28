import argparse

from mvits.trainer import MVITSTrainer
from mvits.utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="configs/config.yml", help='配置文件路径')
parser.add_argument('-m', '--model_dir', type=str, default="models", help='训练保存模型的路径')
parser.add_argument('-e', '--epochs', type=int, default=10000, help='训练轮数')
parser.add_argument('-r', '--resume_model', type=str, default=None, help='恢复训练模型路径')
parser.add_argument('-p', '--pretrained_model', type=str, default=None, help='预训练模型路径')
args = parser.parse_args()
print_arguments(args=args)

trainer = MVITSTrainer(configs=args.config)

trainer.train(epochs=args.epochs, model_dir=args.model_dir,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
