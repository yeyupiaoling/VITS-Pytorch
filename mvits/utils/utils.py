import os

import torch
from tqdm import tqdm

from loguru import logger
from mvits import __version__
from mvits.text import clean_text_


def load_checkpoint(checkpoint_path, model, optimizer=None, drop_speaker_emb=False, is_pretrained=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict.get('epoch', 0)
    version = checkpoint_dict.get('version', 0)
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            if k == 'emb_g.weight' or k == 'enc_p.emb.weight':
                if drop_speaker_emb:
                    continue
                new_state_dict[k] = v
                v[:saved_state_dict[k].shape[0], :] = saved_state_dict[k]
                new_state_dict[k] = v
            else:
                new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=not is_pretrained)
    else:
        model.load_state_dict(new_state_dict, strict=not is_pretrained)
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch}, version {version})")
    return model, optimizer, learning_rate, epoch


# 保存模型
def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path, speakers, text_cleaner):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict, 'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'epoch': epoch, 'text_cleaner': text_cleaner, 'speakers': speakers, 'version': __version__,
                'learning_rate': learning_rate}, checkpoint_path)
    logger.info(f"Save checkpoint '{checkpoint_path}' (epoch {epoch})")


# 处理文本数据
def preprocess(data_anno, list_path, text_cleaner, speaker2id):
    # 生成音素数据
    cleaned_new_annos = []
    for i, line in enumerate(tqdm(data_anno)):
        path, speaker, txt = line.split("|")
        if len(txt) > 150: continue
        cleaned_text = clean_text_(txt, text_cleaner)
        cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
        cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

    # 写入到训练和测试列表中
    with open(list_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(cleaned_new_annos):
            f.write(line)


def plot_spectrogram_to_numpy(spectrogram):
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def print_arguments(args=None, configs=None):
    if args:
        logger.info("----------- 额外配置参数 -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
    if configs:
        logger.info("----------- 配置文件参数 -----------")
        for arg, value in sorted(configs.items()):
            if isinstance(value, dict):
                logger.info(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            logger.info("\t\t%s: %s" % (a1, v1))
                    else:
                        logger.info("\t%s: %s" % (a, v))
            else:
                logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst
