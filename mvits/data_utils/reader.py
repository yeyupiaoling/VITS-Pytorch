import os
import random

import torch
import torch.utils.data
import torchaudio

from mvits.data_utils.mel_processing import spectrogram_torch
from mvits.text import text_to_sequence, cleaned_text_to_sequence
from mvits.models import commons


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audio_paths_sid_text, configs, symbols):
        self.audio_paths_sid_text = self.load_filepaths_and_text(audio_paths_sid_text)
        self.text_cleaner = configs.text_cleaner
        self.max_wav_value = configs.max_wav_value
        self.sampling_rate = configs.sampling_rate
        self.filter_length = configs.filter_length
        self.hop_length = configs.hop_length
        self.win_length = configs.win_length

        self.cleaned_text = configs.get("cleaned_text", False)

        self.add_blank = configs.add_blank
        self.min_text_len = configs.get("min_text_len", 1)
        self.max_text_len = configs.get("max_text_len", 190)
        self.symbols = symbols

        random.seed(1234)
        random.shuffle(self.audio_paths_sid_text)
        self._filter()

    def _filter(self):
        audio_paths_sid_text_new = []
        lengths = []
        for audio_path, sid, text in self.audio_paths_sid_text:
            if self.min_text_len <= len(text) <= self.max_text_len:
                audio_paths_sid_text_new.append([audio_path, sid, text])
                lengths.append(os.path.getsize(audio_path) // (2 * self.hop_length))
        self.audio_paths_sid_text = audio_paths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audio_path_sid_text):
        # separate filename, speaker_id and text
        audio_path, sid, text = audio_path_sid_text[0], audio_path_sid_text[1], audio_path_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audio_path)
        sid = self.get_sid(sid)
        return text, spec, wav, sid

    def get_audio(self, filename):
        audio_norm, sampling_rate = torchaudio.load(filename)
        audio_norm = torchaudio.functional.resample(audio_norm, orig_freq=sampling_rate, new_freq=self.sampling_rate)
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                 self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)
        spec = spec.squeeze(0)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text, self.symbols)
        else:
            text_norm = text_to_sequence(text, self.symbols, self.text_cleaner)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.tensor(text_norm, dtype=torch.long)
        return text_norm

    @staticmethod
    def get_sid(sid):
        sid = torch.tensor([int(sid)], dtype=torch.long)
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audio_paths_sid_text[index])

    def __len__(self):
        return len(self.audio_paths_sid_text)

    @staticmethod
    def load_filepaths_and_text(filename, split="|"):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
        return filepaths_and_text
