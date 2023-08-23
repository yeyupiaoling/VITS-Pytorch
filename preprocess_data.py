import argparse
import json

from tqdm import tqdm

from mvits.text import clean_text_

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.json", help='配置文件路径')
parser.add_argument("--data_list", default="dataset/sampled_audio4ft.txt", help='数据列表路径')
args = parser.parse_args()


# 读取数据列表
with open(args.data_list, 'r', encoding='utf-8') as f:
    new_annos = f.readlines()

# 获取说话人名称
speakers = []
for line in new_annos:
    path, speaker, text = line.split("|")
    if speaker not in speakers:
        speakers.append(speaker)
assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure."
# 读取原配置文件参数
with open(args.config, 'r', encoding='utf-8') as f:
    hps = json.load(f)

# 把说话人名称转换为ID
speaker2id = {}
for i, speaker in enumerate(speakers):
    speaker2id[speaker] = i
# 更新配置参数
hps['data']["n_speakers"] = len(speakers)
hps['speakers'] = speaker2id
# 写入到新的配置文件里面
with open("configs/config.json", 'w', encoding='utf-8') as f:
    json.dump(hps, f, indent=2)

# 生成音素数据
cleaned_new_annos = []
for i, line in enumerate(tqdm(new_annos)):
    path, speaker, txt = line.split("|")
    if len(txt) > 150: continue
    cleaned_text = clean_text_(txt, hps['data']['text_cleaners']).replace("[ZH]", "")
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
print("数据处理完成！")
