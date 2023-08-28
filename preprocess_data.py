import argparse

from mvits.trainer import MVITSTrainer
from mvits.utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yml", help='配置文件路径')
parser.add_argument("--train_data_list", default="dataset/bznsyp.txt", help='训练数据列表路径')
parser.add_argument("--val_data_list", default=None, help='测试数据列表路径，如果没有将从训练数据列表中划分验证集')
args = parser.parse_args()
print_arguments(args=args)

trainer = MVITSTrainer(configs=args.config, is_train=False)

trainer.preprocess_data(train_data_list=args.train_data_list,
                        val_data_list=args.val_data_list)
