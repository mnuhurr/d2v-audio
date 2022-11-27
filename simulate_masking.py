import torch
from models.masking import mask_tokens

def main():
    n_rounds = 100
    batch_size = 100
    batch_len = 16
    dim = 1
    
    p = 0.15
    masklen = 5

    total_masked = 0

    for r in range(n_rounds):
        x = torch.zeros(batch_size, batch_len, dim)
        xm, mask = mask_tokens(x, p_masking=p, masking_length=masklen)

        total_masked += torch.sum(mask < 0)

    print(total_masked / (n_rounds * batch_size * batch_len))


if __name__ == '__main__':
    main()
