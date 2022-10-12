from abc import ABC, abstractmethod
from importlib import import_module
from typing import Iterable, Dict, Any

import torch

METRICS = {
    "ner_f1": "edgar.trainer.metrics.fscore.NERF1",
    "ner_f1_adjusted": "edgar.trainer.metrics.fscore_adjusted.NERF1Adjusted",
    "re_f1": "edgar.trainer.metrics.fscore.REF1",
    "re_f1_adjusted": "edgar.trainer.metrics.fscore_adjusted.REF1Adjusted",
}


class Metric(ABC):
    @classmethod
    def from_config(cls, type_: str, *args, **kwargs):
        try:
            callable_path = METRICS[type_]
            parts = callable_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'Metric "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)
