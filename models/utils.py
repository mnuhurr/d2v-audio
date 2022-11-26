import torch


def model_size(model: torch.nn.Module) -> int:
    num_params = sum(param.numel() for param in model.parameters())
    return num_params

