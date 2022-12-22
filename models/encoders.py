import torch
import torch.nn.functional as F

from typing import List, Tuple


# base class
class ConvEncoder(torch.nn.Module):
    def __init__(self, d_data, d_model, kernel_sizes, strides, paddings):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings

        self.layers = self.build_conv(d_data, d_model)

    def build_conv(self, d_data, d_model):
        modules = []
        for k, (kernel_size, stride, padding) in enumerate(zip(self.kernel_sizes, self.strides, self.paddings)):
            if k == 0:
                in_c = d_data
            else:
                in_c = d_model

            modules.append(torch.nn.Conv1d(in_c, d_model, kernel_size=kernel_size, stride=stride, padding=padding))
            modules.append(torch.nn.GELU())

        return torch.nn.Sequential(*modules)

    def get_conv_params(self) -> Tuple[List[int], List[int], List[int]]:
        return self.kernel_sizes, self.strides, self.paddings


class WaveEncoder(ConvEncoder):
    def __init__(self, 
                 d_model: int = 128, 
                 kernel_sizes: List[int] = None,
                 strides: List[int] = None,
                 paddings: List[int] = None):

        # structure resembling wav2vec
        kernel_sizes = [10, 5, 5, 3, 3, 2, 2] if kernel_sizes is None else kernel_sizes
        strides = [5, 4, 4, 2, 2, 2, 2] if strides is None else strides
        paddings = [0] * len(strides) if paddings is None else paddings

        super().__init__(1, d_model, kernel_sizes=kernel_sizes, strides=strides, paddings=paddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.unsqueeze(1))


class MelEncoder(ConvEncoder):
    def __init__(self, n_mels, d_model: int = 128):
        super().__init__(n_mels, d_model, kernel_sizes=[7, 5, 3], strides=[2, 2, 2], paddings=[1, 1, 1])

    def forward(self, x):
        return self.layers(x)


