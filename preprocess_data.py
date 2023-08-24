import argparse

import yaml
from tqdm import tqdm

from mvits.text import clean_text_
from mvits.utils.logger import setup_logger
from mvits.utils.utils import print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yml", help='配置文件路径')
parser.add_argument("--data_list", default="dataset/sampled_audio4ft.txt", help='数据列表路径')
args = parser.parse_args()
print_arguments(args=args)

logger.info('目前只支持语言：["日本語", "简体中文", "English", "한국어"]')
# 读取数据列表
with open(args.data_list, 'r', encoding='utf-8') as f:
    new_annos = f.readlines()

# 获取说话人名称
speakers = []
for line in new_annos:
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

# 生成音素数据
cleaned_new_annos = []
for i, line in enumerate(tqdm(new_annos)):
    path, speaker, txt = line.split("|")
    if len(txt) > 150: continue
    cleaned_text = clean_text_(txt, configs['data']['text_cleaners'])
    cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
    cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

# 写入到训练和测试列表中
with open("dataset/train.txt", 'w', encoding='utf-8') as f_train, \
        open("dataset/val.txt", 'w', encoding='utf-8') as f_test:
    for i, line in enumerate(cleaned_new_annos):
        if i % 100 == 0:
            f_test.write(line)
        else:
            f_train.write(line)
logger.info("数据处理完成！")
