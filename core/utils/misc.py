import os
import random
from typing import Iterable

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def limit_gpus(gpu_ids: Iterable[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def shorten_string(text, max_length):
    if len(text) <= max_length:
        return text
    else:
        start = text[: max_length // 2].rstrip()
        end = text[-max_length // 2 :].lstrip()
        return f"{start}  ...  {end}"


def get_tensor_size(tensor):
    """Returns the size of a tensor in bytes."""
    return tensor.element_size() * tensor.numel()


def get_nested_tensor_size(nested_tensor):
    """Recursively finds the total size of all tensors in a nested structure."""
    if isinstance(nested_tensor, torch.Tensor):
        return get_tensor_size(nested_tensor)
    elif isinstance(nested_tensor, dict):
        return sum(get_nested_tensor_size(v) for v in nested_tensor.values())
    elif isinstance(nested_tensor, list) or isinstance(nested_tensor, tuple):
        return sum(get_nested_tensor_size(v) for v in nested_tensor)
    else:
        return 0
