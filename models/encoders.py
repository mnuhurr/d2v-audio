import torch
import torch.nn.functional as F

from typing import List, Tuple

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class ResidualUnit(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, kernel_size: int = 7):
        super().__init__()

        self.dilation = dilation

        self.layers = torch.nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, dilation=dilation),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class NormalizedCausalConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1):
        super().__init__()

        self.conv = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.ln = torch.nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv(x))
        x = self.ln(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


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
        super().__init__(n_mels, d_model, kernel_sizes=[3, 3, 3], strides=[1, 2, 2], paddings=[1, 1, 1])

    def forward(self, x):
        return self.layers(x)


