from __future__ import annotations

import logging
from abc import abstractmethod
from importlib import import_module
from typing import Dict

from torch import nn

logger = logging.getLogger(__name__)

ENCODER_models: Dict = {"sentenceEncoder": "edgar.models.encoders.SentenceEncoder",
                        "edgarW2V":  "edgar.models.encoders.EdgarW2VEncoder",
                        "glove": "edgar.models.encoders.GloveEncoder",
                        "tfIdf": "to do"}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict):
        raise NotImplementedError

    @classmethod
    def from_config(cls,
                    encoder_type_: str,
                    *args,
                    **kwargs) -> Encoder:
        try:
            callable_path = ENCODER_models[encoder_type_]
            parts = callable_path.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'Encoder "{encoder_type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)
