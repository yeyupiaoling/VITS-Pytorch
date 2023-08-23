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

    def __init__(self, audiopaths_sid_text, hparams, symbols):
        self.audiopaths_sid_text = self.load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.cleaned_text = hparams.get("cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = hparams.get("min_text_len", 1)
        self.max_text_len = hparams.get("max_text_len", 190)
        self.symbols = symbols

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return text, spec, wav, sid

    def get_audio(self, filename):
        # audio, sampling_rate = load_wav_to_torch(filename)
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} {} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))
        # audio_norm = audio / self.max_wav_value if audio.max() > 10 else audio
        # audio_norm = audio_norm.unsqueeze(0)
        audio_norm, sampling_rate = torchaudio.load(filename, frame_offset=0, num_frames=-1, normalize=True,
                                                    channels_first=True)
        # spec_filename = filename.replace(".wav", ".spec.pt")
        # if os.path.exists(spec_filename):
        #     spec = torch.load(spec_filename)
        # else:
        #     try:
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                 self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)
        spec = spec.squeeze(0)
        # except NotImplementedError:
        #     print("?")
        # spec = torch.squeeze(spec, 0)
        # torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text, self.symbols)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    @staticmethod
    def load_filepaths_and_text(filename, split="|"):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
        return filepaths_and_text
