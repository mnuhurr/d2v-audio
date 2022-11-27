import torch
import torchaudio

from pathlib import Path
from typing import List, Union, Optional


class RawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, filenames: List[Union[str, Path]], sample_rate: int = 16000, max_duration: float = 10.0):
        super().__init__()

        self.filenames = filenames
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        y, sr = torchaudio.load(self.filenames[item])

        # make it mono
        y = torch.mean(y, dim=0)

        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)

        if len(y) > self.max_length:
            offset = torch.randint(y.size(-1) - self.max_length, size=(1,))[0]
            y = y[offset:offset + self.max_length]

        return y

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 filenames: List[Union[str, Path]], 
                 sample_rate: int = 16000, 
                 n_mels: int = 64,
                 n_fft: int = 1024,
                 hop_length: Optional[int] = None,
                 max_length: int = 320,
                 log_mel_scaling: float = 0.1):

        super().__init__()

        self.filenames = filenames
        self.sample_rate = sample_rate
        self.max_length = max_length
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length,
            n_mels=n_mels)

        self.scaling = log_mel_scaling

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        y, sr = torchaudio.load(self.filenames[item])

        # make it mono
        y = torch.mean(y, dim=0)

        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)
    
        mels = self.melspec(y)
        # (n_mels, t)

        if mels.size(-1) > self.max_length:
            offset = torch.randint(mels.size(-1) - self.max_length, size=(1,))[0]
            mels = mels[:, offset:offset + self.max_length]

        log_mels = self.scaling * torch.log(mels + torch.finfo(torch.float32).eps)

        return log_mels

