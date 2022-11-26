import torch
import torchaudio

from pathlib import Path
from typing import List, Union


class RawAudioDataset(torch.nn.Module):
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

