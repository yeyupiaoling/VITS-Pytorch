import torch
import yaml

from mvits.models import commons
from mvits.models.models import SynthesizerTrn
from mvits.text import text_to_sequence
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
        self.speaker_ids = configs['speakers']
        self.configs = dict_to_object(configs)
        self.symbols = self.configs['symbols']
        # 获取模型
        self.net_g = SynthesizerTrn(len(self.symbols),
                                    self.configs.data.filter_length // 2 + 1,
                                    self.configs.train.segment_size // self.configs.data.hop_length,
                                    n_speakers=self.configs.data.n_speakers,
                                    **self.configs.model).to(self.device)
        self.net_g.eval()
        load_checkpoint(model_path, self.net_g, None)
        self.language_marks = {"日本語": "[JA]", "简体中文": "[ZH]", "English": "[EN]", "한국어": "[KO]"}
        logger.info(f'支持说话人：{list(self.speaker_ids.keys())}')

    @staticmethod
    def get_text(text, hps, is_symbol):
        text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
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
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            sid = torch.LongTensor([speaker_id]).to(self.device)
            audio = \
                self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                 length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio, self.configs.data.sampling_rate
