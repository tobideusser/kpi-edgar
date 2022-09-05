from importlib import import_module
import inspect
from typing import Dict, Any, Optional

import torch


LR_SCHEDULERS = {
    "exponential": "torch.optim.lr_scheduler.ExponentialLR",
    "lin_warmup": "transformers.get_linear_schedule_with_warmup",
}


class LearningRateScheduler:
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, interval: str = "epoch") -> None:
        self.lr_scheduler = lr_scheduler
        self.use_metric = True if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else False
        self.interval = interval

    def get_values(self):
        return self.lr_scheduler.get_last_lr()

    def step(self, metric: Optional[float] = None) -> None:
        if self.use_metric and metric is not None:
            self.lr_scheduler.step(metric=metric)
        self.lr_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)

    @classmethod
    def from_config(cls, type_: str, optimizer: torch.optim.Optimizer, **kwargs):
        try:
            callable_path = LR_SCHEDULERS[type_]
            parts = callable_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'{cls.__name__} "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)

        expected_scheduler_args = inspect.signature(class_).parameters.keys()
        scheduler_kwargs = {name: value for name, value in kwargs.items() if name in expected_scheduler_args}

        interval = kwargs.pop("interval", "epoch")
        lr_scheduler = class_(optimizer=optimizer, **scheduler_kwargs)

        return cls(lr_scheduler=lr_scheduler, interval=interval)
