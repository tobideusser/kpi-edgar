import multiprocessing
from typing import List, Optional

import torch


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
