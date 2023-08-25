import os

import soundfile
from tqdm import tqdm


def create_aihell3_list(aishell2_dir, data_type='train'):
    with open(os.path.join(aishell2_dir, data_type, 'content.txt'), encoding='utf-8') as f:
        lines = f.readlines()
    labels_dict = dict()

    for line in lines:
        line = line.replace('\n', '')
        name, labels = line.split('\t')
        labels = labels.split(' ')
        label = ''.join([labels[i] for i in range(0, len(labels), 2)])
        labels_dict[name] = label

    path = os.path.join(aishell2_dir, data_type, 'wav')
    with open(f"dataset/aishell3_{data_type}.txt", 'w', encoding='utf-8') as fw:
        for d in tqdm(os.listdir(path)):
            for name in os.listdir(os.path.join(path, d)):
                audio_path = os.path.join(path, d, name).replace('\\', '/')
                # sample, sr = soundfile.read(audio_path)
                # duration = float(sample.shape[0] / sr)
                # if duration < 2 or duration > 10: continue
                label = labels_dict[name].replace(' ', '')
                # if len(label) < 10: continue
                text = f'{audio_path}|{d}|[ZH]{label}[ZH]\n'
                fw.write(text)


def create_bznsyp(data_dir):
    with open(os.path.join(data_dir, 'ProsodyLabeling', '000001-010000.txt'), encoding='utf-8') as f:
        lines = f.readlines()
    labels_dict = dict()

    for i, line in enumerate(lines):
        if i % 2 != 0:continue
        line = line.replace('\n', '')
        name, label = line.split('\t')
        label = label.replace('#1', '').replace('#2', '').replace('#3', '').replace('#4', '')
        labels_dict[name] = label

    path = os.path.join(data_dir, 'Wave')
    with open(f"dataset/bznsyp.txt", 'w', encoding='utf-8') as fw:
        for f in tqdm(os.listdir(path)):
            if not f.endswith('.wav'):continue
            audio_path = os.path.join(path, f).replace('\\', '/')
            # sample, sr = soundfile.read(audio_path)
            # duration = float(sample.shape[0] / sr)
            # if duration < 2 or duration > 10: continue
            name = f[:-4]
            label = labels_dict[name].replace(' ', '')
            # if len(label) < 10: continue
            text = f'{audio_path}|标准女声|[ZH]{label}[ZH]\n'
            fw.write(text)


if __name__ == '__main__':
    # create_aihell3_list(aishell2_dir='dataset/data_aishell3/', data_type='train')
    # create_aihell3_list(aishell2_dir='dataset/data_aishell3/', data_type='test')
    create_bznsyp('dataset/BZNSYP')
