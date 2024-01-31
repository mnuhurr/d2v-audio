import torch

from .encoders import WaveEncoder
from .encoders import MelEncoder
from .encoders import ConvNeXtEncoder
from .transformer import TransformerEncoder

from typing import Tuple, Optional, Any, Union


class D2VEncoder(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 d_ff: Optional[int] = None,
                 max_sequence_length: int = 1024,
                 n_mels: Optional[int] = None,
                 p_masking: float = 0.065,
                 masking_length: int = 10,
                 d_encoder: Optional[int] = None,
                 codebook_size: int = 1024):

        super().__init__()

        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.encoder = ConvNeXtEncoder(n_channels=32, d_model=d_model)

        """
        d_encoder = d_encoder if d_encoder is not None else d_model

        if n_mels is None or n_mels == 0:
            self.encoder = WaveEncoder(d_encoder)
        else:
            self.encoder = MelEncoder(n_mels, d_encoder)
        """

        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_ff=d_ff,
            n_heads=n_heads,
            max_sequence_length=max_sequence_length)

        self.projection = torch.nn.Linear(d_model, d_model)

        self.mask_token = torch.nn.Parameter(0.1 * torch.randn(d_model))

        self.register_buffer('p_masking', torch.tensor(p_masking), persistent=False)
        self.register_buffer('masking_length', torch.tensor(masking_length), persistent=False)

    def forward(self,
                x: torch.Tensor,
                input_mask: Optional[torch.Tensor] = None,
                mode: str = 'encoder') -> Union[torch.Tensor, Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, Any]]:
        assert mode in ['encoder', 'student', 'teacher']

        # extract tokens
        x = self.encoder(x)
        #x = x.permute(0, 2, 1)

        if input_mask is not None:
            #input_mask = contract_mask(input_mask, *self.encoder.get_conv_params())
            input_mask = torch.nn.functional.max_pool1d(input_mask, self.encoder.downsampling_factor())

        training_mask = None
        if mode == 'student':
            x, training_mask = self.mask_tokens(x)

        # get transformer outputs
        x, attn_weights = self.transformer(x, mask=input_mask)

        # if we are in the student mode also compute the projection and add it to the same output list
        if mode == 'student':
            x_proj = self.projection(x[-1])
            x.append(x_proj)

        # return attention weights for other modes?
        if mode == 'encoder':
            return x, attn_weights, input_mask
        elif mode == 'student':
            return x, training_mask, input_mask
        elif mode == 'teacher':
            return x, input_mask

    def mask_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mask input tokens and generate corresponding mask. assume
            x.shape = [batch, t, d_model]
            mask_token.shape = [d_model]

        output mask has shape [batch, t] and contains zero for unmasked positions and -inf for masked positions

        :param x: input data
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

            x[i, j:j + self.masking_length, :] = self.mask_token

            mask[i, j:j + self.masking_length] = -float('inf')

        return x, mask.to(self.p_masking.device)
