import torch

from typing import Optional, Tuple


def mask_tokens(x: torch.Tensor,
                mask_token: Optional[torch.Tensor] = None,
                p_masking: float = 0.065,
                masking_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mask input tokens and generate corresponding mask. assume
        x.shape = [batch, t, d_model]
        mask_token.shape = [d_model]

    output mask has shape [batch, t] and contains zero for unmasked positions and -inf for masked positions

    :param x: input data
    :param mask_token: optional mask token to be inserted in x for masked positions
    :param p_mask: probability for one time step to be the starting position of a mask
    :param mask_length: number of tokens to mask out for every starting pos
    :return: masked x, mask
    """

    batch_size, t = x.shape[:2]

    mask = torch.zeros(batch_size, t)
    start_pos = torch.rand(batch_size, t) < p_masking
    n_starts = start_pos.sum()
    start_ind = torch.where(start_pos)

    # make a copy of x before to keep the original intact
    x = x.clone()

    for k in range(n_starts):
        i = start_ind[0][k]
        j = start_ind[1][k]

        if mask_token is not None:
            x[i, j:j + masking_length, :] = mask_token

        mask[i, j:j + masking_length] = -float('inf')

    return x, mask


def foo():
    x = torch.randn(2, 24, 8)
    mt = torch.ones(8)
    xm, mask = mask_tokens(x, mt)

    print(x - xm)


if __name__ == '__main__':
    foo()