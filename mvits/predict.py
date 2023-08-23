import torch
from torch import no_grad, LongTensor

from mvits.models import commons
from mvits.models.models import SynthesizerTrn
from mvits.text import text_to_sequence
from mvits.utils.logger import setup_logger
from mvits.utils.utils import load_checkpoint, get_hparams_from_file

logger = setup_logger(__name__)


class MVITSPredictor:
    def __init__(self, config_path, model_path, use_gpu=True):
        self.device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
        self.hps = get_hparams_from_file(config_path)
        self.speaker_ids = self.hps.speakers
        # 获取模型
        self.net_g = SynthesizerTrn(len(self.hps.symbols),
                                    self.hps.data.filter_length // 2 + 1,
                                    self.hps.train.segment_size // self.hps.data.hop_length,
                                    n_speakers=self.hps.data.n_speakers,
                                    **self.hps.model).to(self.device)
        self.net_g.eval()
        load_checkpoint(model_path, self.net_g, None)
        self.language_marks = {"Japanese": "", "日本語": "[JA]", "简体中文": "[ZH]", "English": "[EN]", "Mix": ""}
        logger.info(f'支持说话人：{list(self.speaker_ids.keys())}')

    @staticmethod
    def get_text(text, hps, is_symbol):
        text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def generate(self, text, spk, language, noise_scale=0.667, noise_scale_w=0.6, speed=1):
        assert spk in self.speaker_ids.keys()
        assert language in self.language_marks.keys()
        # 输入到模型的文本
        text = self.language_marks[language] + text + self.language_marks[language]
        speaker_id = self.speaker_ids[spk]
        stn_tst = self.get_text(text, self.hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            sid = LongTensor([speaker_id]).to(self.device)
            audio = \
                self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                 length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio, self.hps.data.sampling_rate
