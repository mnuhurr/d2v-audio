import torch
import torch.nn.functional as F

from .encoders import WaveEncoder
from .transformer import TransformerEncoder

from typing import Tuple, Optional


class D2VEncoder(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int,
                 d_ff: int,
                 n_heads: int,
                 max_sequence_length: int,
                 p_masking: float = 0.065,
                 masking_length: int = 10):

        super().__init__()

        self.encoder = WaveEncoder(d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_ff=d_ff,
            n_heads=n_heads,
            max_sequence_length=max_sequence_length)

        self.register_buffer('mask_token', torch.randn(d_model))

        self.register_buffer('p_masking', torch.tensor(p_masking), persistent=False)
        self.register_buffer('masking_length', torch.tensor(masking_length), persistent=False)

    def forward(self, x: torch.Tensor, masking: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.encoder(x)

        x = x.permute(0, 2, 1)

        mask = None
        if masking:
            x, mask = self.mask_tokens(x, mask_token=self.mask_token)

        x, attn_weights = self.transformer(x)

        return x, mask

    def mask_tokens(self,
                    x: torch.Tensor,
                    mask_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mask input tokens and generate corresponding mask. assume
            x.shape = [batch, t, d_model]
            mask_token.shape = [d_model]

        output mask has shape [batch, t] and contains zero for unmasked positions and -inf for masked positions

        :param x: input data
        :param mask_token: optional mask token to be inserted in x for masked positions
        :return: masked x, mask
        """

        batch_size, t = x.shape[:2]

        mask = torch.zeros(batch_size, t)
        start_pos = torch.rand(batch_size, t).to(self.p_masking.device) < self.p_masking
        n_starts = start_pos.sum()
        start_ind = torch.where(start_pos)

        # make a copy of x before to keep the original intact
        x = x.clone()

        for k in range(n_starts):
            i = start_ind[0][k]
            j = start_ind[1][k]

            if mask_token is not None:
                x[i, j:j + self.masking_length, :] = mask_token

            mask[i, j:j + self.masking_length] = -float('inf')

        return x, mask.to(self.p_masking.device)
