import torch
import torch.nn.functional as F


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


class WaveEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()

        self.layers = torch.nn.Sequential(
            NormalizedCausalConvBlock(1, d_model, kernel_size=10, stride=5),
            NormalizedCausalConvBlock(d_model, d_model, kernel_size=7, stride=4),
            NormalizedCausalConvBlock(d_model, d_model, kernel_size=7, stride=4),
            NormalizedCausalConvBlock(d_model, d_model, kernel_size=5, stride=4),
            NormalizedCausalConvBlock(d_model, d_model, kernel_size=3, stride=2),
            NormalizedCausalConvBlock(d_model, d_model, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.unsqueeze(1))


def foo():
    from utils import model_size

    x = torch.randn(2, 10*16000)
    wenc = WaveEncoder(256)

    y = wenc(x)
    print(y.shape)

    print(model_size(wenc))


if __name__ == '__main__':
    foo()
