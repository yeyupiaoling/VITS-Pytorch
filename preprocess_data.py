import argparse
import os
import random

import yaml
from tqdm import tqdm

from mvits.text import clean_text_
from mvits.utils.logger import setup_logger
from mvits.utils.utils import print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yml", help='配置文件路径')
parser.add_argument("--train_data_list", default="dataset/aishell3_train.txt", help='训练数据列表路径')
parser.add_argument("--val_data_list", default=None, help='测试数据列表路径，如果没有将从训练数据列表中划分验证集')
args = parser.parse_args()
print_arguments(args=args)

logger.info('目前只支持语言：["日本語", "简体中文", "English", "한국어"]')


def preprocess_data(data_anno, list_path, text_cleaners, speaker2id):
    # 生成音素数据
    cleaned_new_annos = []
    for i, line in enumerate(tqdm(data_anno)):
        path, speaker, txt = line.split("|")
        if len(txt) > 150: continue
        cleaned_text = clean_text_(txt, text_cleaners)
        cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
        cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

    # 写入到训练和测试列表中
    with open(list_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(cleaned_new_annos):
            f.write(line)


def main():
    assert os.path.join(args.train_data_list), f"训练数据列表文件不存在：{args.train_data_list}"
    # 读取数据列表
    with open(args.train_data_list, 'r', encoding='utf-8') as f:
        train_anno = f.readlines()
    if args.val_data_list is not None and os.path.join(args.val_data_list):
        with open(args.val_data_list, 'r', encoding='utf-8') as f:
            val_anno = f.readlines()
    else:
        logger.warning(f'测试数据列表文件不存在，将从训练数据列表中按100:1划分验证集')
        temp_indexes = list(range(len(train_anno)))
        choice_indexes = random.choices(temp_indexes, k=len(train_anno) // 100)
        val_anno = [train_anno[i] for i in choice_indexes]
        train_anno = [train_anno[i] for i in set(temp_indexes) - set(choice_indexes)]

    # 获取说话人名称
    speakers = []
    for line in train_anno:
        path, speaker, text = line.split("|")
        if speaker not in speakers:
            speakers.append(speaker)
    assert (len(speakers) != 0), "没有说话人数据！"
    # 读取原配置文件参数
    with open(args.config, 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 把说话人名称转换为ID
    speaker2id = {}
    for i, speaker in enumerate(speakers):
        speaker2id[speaker] = i
    # 更新配置参数
    configs['data']["n_speakers"] = len(speakers)
    configs['speakers'] = speaker2id
    # 写入到新的配置文件里面
    with open('configs/config.yml', 'w', encoding='utf-8') as f:
        yaml_datas = yaml.dump(configs, indent=2, sort_keys=False, allow_unicode=True)
        f.write(yaml_datas)

    preprocess_data(data_anno=train_anno,
                    list_path='dataset/train.txt',
                    text_cleaners=configs['data']['text_cleaners'],
                    speaker2id=speaker2id)
    preprocess_data(data_anno=val_anno,
                    list_path='dataset/val.txt',
                    text_cleaners=configs['data']['text_cleaners'],
                    speaker2id=speaker2id)
    logger.info("数据处理完成！")


if __name__ == '__main__':
    main()
