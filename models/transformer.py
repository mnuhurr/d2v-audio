
import math
import torch
import torch.nn.functional as F

from typing import Optional, Tuple, Dict, List


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume x.shape is (batch, t, d_model)
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = torch.nn.LayerNorm(d_model)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.ff_ln = torch.nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. self attention
        x = self.attn_ln(x)
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out

        # 2. ff network
        x = x + self.ff(self.ff_ln(x))

        return x, attn_weights.detach()


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = torch.nn.LayerNorm(d_model)

        self.cross_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_ln = torch.nn.LayerNorm(d_model)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.ff_ln = torch.nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                xa: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                xa_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. self attention
        x = self.attn_ln(x)
        attn_out, self_attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out

        # 2. cross attention
        x = self.cross_attn_ln(x)
        attn_out, cross_attn_weights = self.cross_attn(query=x, key=xa, value=xa, key_padding_mask=xa_mask)
        x = x + attn_out

        # 3. ff network
        x = x + self.ff(self.ff_ln(x))

        return x, self_attn_weights.detach(), cross_attn_weights.detach()


class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_ff: int, n_heads: int, max_sequence_length: int):
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_sequence_length)
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = self.positional_encoding(x)

        # return list of
        layer_outputs = []
        attn_weights = []
        for k, layer in enumerate(self.layers):
            x, w = layer(x, mask=mask)
            layer_outputs.append(x)
            attn_weights.append(w)

        return layer_outputs, attn_weights


def foo():
    x = torch.randn(2, 12, 8)
    y = torch.randn(2, 14, 8)

    mask = -float('inf') * torch.ones(2, x.size(1))
    mask[:, :10] = 0

    xmask = -float('inf') * torch.ones(2, y.size(1))
    xmask[:, :12] = 0

    enc = TransformerEncoder(8, 4, 32, 2, 64)
    y, w = enc(x, mask=mask)

    print(torch.stack(y).shape)


if __name__ == '__main__':
    foo()
