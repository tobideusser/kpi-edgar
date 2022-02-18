import multiprocessing
import os
import random
from typing import List, Optional, Union, Sequence, Dict

import numpy as np
import torch


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_DEVICE: Optional[torch.device] = None
_SEED: Optional[int] = None


def get_balanced_devices(count: Optional[int] = None,
                         use_cuda: bool = True,
                         cuda_ids: Optional[List[int]] = None) -> List[str]:
    count = count if count is not None else multiprocessing.cpu_count()
    if use_cuda and torch.cuda.is_available():
        if cuda_ids is not None:
            devices = [f'cuda:{id_}' for id_ in cuda_ids]
        else:
            devices = [f'cuda:{id_}' for id_ in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def get_padding_mask(seqlens: List[int], device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    mask = torch.zeros((len(seqlens), max(seqlens)), dtype=torch.bool, device=device)
    for i, l in enumerate(seqlens):
        mask[i, :l] = 1
    return mask


def nan_safe_tensor_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result


def clamp_tensor(tensor: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    """
    Supports sparse and dense tensors.
    Returns a tensor with values clamped between the provided minimum and maximum,
    without modifying the original tensor.
    """
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()
        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)


def set_device(device: str):
    global _DEVICE
    _DEVICE = torch.device(device)


def get_device() -> torch.device:
    return _DEVICE


def set_seed_number(seed: int):
    global _SEED
    _SEED = seed


def set_seeds():
    torch.manual_seed(_SEED)
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.cuda.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(_SEED)


def detach_batch(batch: Dict) -> Dict:
    # Detach tensors and move to cpu
    return {name: (result.detach().cpu() if isinstance(result, torch.Tensor) else result)
            for name, result in batch.items()}


def argsort(seq: Sequence) -> List[int]:
    return sorted(range(len(seq)), key=seq.__getitem__)


def ceildiv(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    return -(a // -b)
