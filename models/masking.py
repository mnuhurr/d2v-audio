import torch

from typing import Optional, Tuple, List


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


def simulate_masking(p: float,
                     mask_len: int,
                     n_rounds: int = 1000,
                     batch_size: int = 32,
                     num_timesteps: int = 64) -> float:
    """
    generate random masks to estimate the amount of masked tokens

    :param p: masking probability
    :param mask_len: masking length
    :param n_rounds: number of simulation rounds
    :param batch_size: batch size
    :param num_timesteps: (batch) data lenght
    :return: fraction of masked tokens
    """
    dim = 1

    total_masked = 0
    for r in range(n_rounds):
        x = torch.zeros(batch_size, num_timesteps, dim)
        xm, mask = mask_tokens(x, p_masking=p, masking_length=mask_len)

        total_masked += torch.sum(mask < 0)

    return total_masked / (n_rounds * batch_size * num_timesteps)


def contract_mask(mask: torch.Tensor, kernel_sizes: List[int], strides: List[int], paddings: List[int]) -> torch.Tensor:
    """
    shrink mask according to the given list of convolution network parameters. assume masked out positions are
    marked with -inf, and positions containing actual information are marked with 0.

    :param mask: original mask
    :param kernel_sizes: list of kernel sizes in the token encoder
    :param strides: list of strides in the token encoder
    :param paddings: list of paddings in the token encoder
    :return: shrinked mask
    """

    for kernel_size, stride, padding in zip(kernel_sizes, strides, paddings):
        mask = torch.nn.functional.max_pool1d(mask, kernel_size=kernel_size, stride=stride, padding=padding)

    return mask


