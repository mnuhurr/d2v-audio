"""
ConvNeXt, original paper & code:
 - https://arxiv.org/abs/2201.03545
 - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

major differences to the original:
 - stem layer is now downsampling by 2 instead of 4
"""
import torch

from typing import List, Tuple


class Scaling(torch.nn.Module):
    def __init__(self, mu: float = 0.0, std: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([mu, std]), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return (x - self.weight[0]) / self.weight[1]


class ConvLayerNorm(torch.nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class DropPath(torch.nn.Module):
    """stochastic depth/drop path"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep_prob)
        if self.scale_by_keep:
            mask.div_(keep_prob)

        return mask * x


class Downsampling(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2, eps: float = 1e-6):
        super().__init__(*[
            ConvLayerNorm(in_channels, eps=eps),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=factor, stride=factor)
        ])


class Block(torch.nn.Module):
    def __init__(self, n_channels: int, kernel_size: int = 7, exp_factor: int = 4, eps: float = 1e-6,
                 layer_scale_init_value: float = 1e-6, drop_path: float = 0.0):
        super().__init__()

        padding = (kernel_size - 1) // 2
        d_hid = exp_factor * n_channels

        self.depthwise = torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=n_channels)

        self.norm = ConvLayerNorm(n_channels, eps=eps)
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, d_hid, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(d_hid, n_channels, kernel_size=1))

        self.gamma = torch.nn.Parameter(layer_scale_init_value * torch.ones(1, n_channels, 1, 1), requires_grad=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.norm(y)
        y = self.pointwise(y)
        y = self.gamma * y
        return x + self.drop_path(y)


class ConvNeXtish(torch.nn.Module):
    def __init__(self, d_embedding: int, depths: List[int] = [3, 3, 9, 3], n_channels: int = 64,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()

        dims = [n_channels * 2**k for k in range(len(depths))]

        self.scaling = Scaling(mu=-0.5, std=6.0)
        self.downsamples = torch.nn.ModuleList()
        self.stages = torch.nn.ModuleList()

        stem = torch.nn.Sequential(
            # torch.nn.Conv2d(1, n_channels, kernel_size=3, padding=1),
            torch.nn.Conv2d(1, n_channels, kernel_size=2, stride=2),
            ConvLayerNorm(n_channels),
        )

        self.downsamples.append(stem)
        for k in range(len(dims) - 1):
            self.downsamples.append(Downsampling(dims[k], dims[k + 1]))

        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        c = 0
        for k in range(len(depths)):
            stage = torch.nn.Sequential(*[
                Block(dims[k], drop_path=dp_rates[c]) for _ in range(depths[k])
            ])
            self.stages.append(stage)
            c += depths[k]

        self.ln = torch.nn.LayerNorm(dims[-1])

        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dims[-1], d_embedding)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            m.weight.data.normal_(0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.clip_embedding(x)
        x = self.embeddings(x)
        x = self.head(x)
        return x

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """embeddings containing temporal axis"""
        # x: (batch, f, t)
        x = self.scaling(x)
        x = x.unsqueeze(1)

        for k in range(len(self.stages)):
            x = self.downsamples[k](x)
            x = self.stages[k](x)

        # x: (batch, c, f', t')

        # average out the feature dimension
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        # x: (batch, t', c)
        return x

    def clip_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """squeeze in the temporal axis and return a single vector for the input audio"""
        x = self.embeddings(x)
        # these two are in different order in the original? (pooling is done with one averaging)
        x = self.ln(x)
        x = x.mean(dim=1)
        return x


def foo():
    from utils import model_size
    model = ConvNeXtish(d_embedding=16, n_channels=32, depths=[3, 3, 27, 3])
    x = torch.randn(4, 64, 200)
    y = model(x)
    print(x.shape, y.shape)
    print(model_size(model) / 1e6)

    # print(model)

if __name__ == '__main__':
    foo()
