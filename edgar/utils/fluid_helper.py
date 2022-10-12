import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional

import torch
from fluidml.common import Resource
from fluidml.storage import LocalFileStore, TypeInfo
from rich.logging import RichHandler
from transformers import PreTrainedTokenizerFast


class MyLocalFileStore(LocalFileStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir=base_dir)

        self._type_registry["torch"] = TypeInfo(torch.save, torch.load, "pt", is_binary=True)
        self._type_registry["tokenizer"] = TypeInfo(self._save_tokenizer, self._load_tokenizer, needs_path=True)

    @staticmethod
    def _save_tokenizer(obj: PreTrainedTokenizerFast, path: str):
        obj.save_pretrained(save_directory=path, legacy_format=False)

    @staticmethod
    def _load_tokenizer(path: str) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(path)


@dataclass
class TaskResource(Resource):
    device: Union[str, torch.device]


def configure_logging(level: Union[str, int] = "INFO", log_dir: Optional[str] = None):
    assert level in ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "FATAL", "CRITICAL", 10, 20, 30, 40, 50]
    logger = logging.getLogger()
    formatter = logging.Formatter("%(processName)-13s%(message)s")
    stream_handler = RichHandler(rich_tracebacks=True, tracebacks_extra_lines=2, show_path=False)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    if log_dir is not None:
        log_path = os.path.join(log_dir, f"{datetime.now()}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(processName)s - %(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.addHandler(stream_handler)
    logger.setLevel(level)
