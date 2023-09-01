import torch
import yaml

from mvits import LANGUAGE_MARKS
from mvits.models.commons import intersperse
from mvits.models.models import SynthesizerTrn
from mvits.text import text_to_sequence, get_symbols
from mvits.utils.logger import setup_logger
from mvits.utils.utils import load_checkpoint, print_arguments, dict_to_object

logger = setup_logger(__name__)


class MVITSPredictor:
    def __init__(self, configs, model_path, use_gpu=True):
        self.device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        checkpoint_dict = torch.load(model_path, map_location='cpu')
        self.speaker_ids = checkpoint_dict['speakers']
        self.text_cleaner = checkpoint_dict['text_cleaner']
        self.symbols = get_symbols(self.text_cleaner)
        # 获取模型
        self.net_g = SynthesizerTrn(len(self.symbols),
                                    self.configs.dataset_conf.filter_length // 2 + 1,
                                    self.configs.train_conf.segment_size // self.configs.dataset_conf.hop_length,
                                    n_speakers=len(self.speaker_ids),
                                    **self.configs.model).to(self.device)
        self.net_g.eval()
        load_checkpoint(model_path, self.net_g, None)
        # 获取支持语言
        if self.text_cleaner in LANGUAGE_MARKS.keys():
            self.language_marks = LANGUAGE_MARKS[self.text_cleaner]
        else:
            raise Exception(f"不支持方法：{self.text_cleaner}")
        logger.info(f'使用文本处理方式为：{self.text_cleaner}')
        logger.info(f'支持说话人：{list(self.speaker_ids.keys())}')

    def get_text(self, text, config, is_symbol):
        text_norm = text_to_sequence(text, self.symbols, [] if is_symbol else self.text_cleaner)
        if config.dataset_conf.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.tensor(text_norm, dtype=torch.long)
        return text_norm

    def generate(self, text, spk, language, noise_scale=0.667, noise_scale_w=0.6, speed=1):
        assert spk in self.speaker_ids.keys(), f'不存在说话人：{spk}'
        assert language in self.language_marks.keys(), f'不支持语言：{language}'
        # 输入到模型的文本
        text = self.language_marks[language] + text + self.language_marks[language]
        speaker_id = self.speaker_ids[spk]
        stn_tst = self.get_text(text, self.configs, False)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.tensor([stn_tst.size(0)], dtype=torch.long).to(self.device)
            sid = torch.tensor([speaker_id], dtype=torch.long).to(self.device)
            audio = \
                self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                 length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio, self.configs.dataset_conf.sampling_rate
