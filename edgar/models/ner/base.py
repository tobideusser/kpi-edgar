from __future__ import annotations

import logging
from abc import abstractmethod
from importlib import import_module
from typing import Dict

from torch import nn

logger = logging.getLogger(__name__)

NER_DECODER: Dict = {"iobes": "edgar.models.ner.IobesNERDecoder", "span": "edgar.models.ner.SpanNERDecoder"}


class NERDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: Dict):
        raise NotImplementedError

    @classmethod
    def from_config(cls, type_: str, *args, **kwargs) -> NERDecoder:
        try:
            callable_path = NER_DECODER[type_]
            parts = callable_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'NER Decoder "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)
