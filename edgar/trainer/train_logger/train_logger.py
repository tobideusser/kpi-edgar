from abc import ABC, abstractmethod
from argparse import Namespace
from collections import MutableMapping
from importlib import import_module
import json
import os
import numbers
from typing import Dict, Optional, Any, Union, List

import torch
import torch.nn as nn


TRAIN_LOGGERS = {'aim': 'edgar.trainer.train_logger.aim_logger.AimLogger',
                 'tensorboard': 'edgar.trainer.train_logger.tensorboard_logger.TensorBoardLogger',
                 'wandb': 'edgar.trainer.train_logger.wandb_logger.WandbLogger'}


class TrainLogger(ABC):
    """
    Base class for experiment loggers.
    """

    def __init__(self,
                 log_dir: str,
                 len_logged_preds_trgs: Optional[Union[int, str]] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        if type(len_logged_preds_trgs) is str:
            if len_logged_preds_trgs == "all":
                self.len_logged_preds_trgs = float("inf")
            else:
                self.len_logged_preds_trgs = None
        else:
            self.len_logged_preds_trgs = len_logged_preds_trgs
        self.best_preds = []
        self.current_preds = []
        self.trgs = [[]]

    @abstractmethod
    def log_samples(
            self,
            batch_output: Dict[str, Any],
            metrics: Dict[str, Any],
            src_tables: List[str],
            prefix: str,
            step: Optional[int] = None
    ):
        """
        Records samples of results and targets.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Records metrics.
        This method logs metrics as as soon as it received them.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        raise NotImplementedError

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None):
        """
        Record hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def watch(self, model: nn.Module, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_graph(self, model: nn.Module, input_array: Optional[torch.Tensor] = None) -> None:
        """
        Record model graph
        """
        raise NotImplementedError

    # def log_memory_usage(self, cpu_memory_usage: Dict[int, int], gpu_memory_usage: Dict[int, int]):
    #     cpu_memory_usage_total = 0.0
    #     for worker, mem_bytes in cpu_memory_usage.items():
    #         memory = mem_bytes / (1024 * 1024)
    #         self.add_train_scalar(f"memory_usage/worker_{worker}_cpu", memory)
    #         cpu_memory_usage_total += memory
    #     self.add_train_scalar("memory_usage/cpu", cpu_memory_usage_total)
    #     for gpu, mem_bytes in gpu_memory_usage.items():
    #         memory = mem_bytes / (1024 * 1024)
    #         self.add_train_scalar(f"memory_usage/gpu_{gpu}", memory)

    # def get_gpu_memory_map() -> Dict[str, int]:
    #     """
    #     Get the current gpu usage.
    #     Return:
    #         A dictionary in which the keys are device ids as integers and
    #         values are memory usage as integers in MB.
    #     """
    #     result = subprocess.run(
    #         [shutil.which("nvidia-smi"), "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
    #         encoding="utf-8",
    #         # capture_output=True,          # valid for python version >=3.7
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
    #         check=True,
    #     )
    #
    #     # Convert lines into a dictionary
    #     gpu_memory = [float(x) for x in result.stdout.strip().split(os.linesep)]
    #     gpu_memory_map = {
    #         f"gpu_id: {gpu_id}/memory.used (MB)": memory for gpu_id, memory in enumerate(gpu_memory)
    #     }
    #     return gpu_memory_map

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @staticmethod
    def _flatten_dict(params: Dict[str, Any], delimiter: str = '/') -> Dict[str, Any]:
        """
        Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
        Args:
            params: Dictionary containing the hyperparameters
            delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
        Returns:
            Flattened dict.
        """

        def _dict_generator(input_dict, prefixes=None):
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, MutableMapping):
                for key, value in input_dict.items():
                    if isinstance(value, (MutableMapping, Namespace)):
                        value = vars(value) if isinstance(value, Namespace) else value
                        for d in _dict_generator(value, prefixes + [key]):
                            yield d
                    else:
                        yield prefixes + [key, value if value is not None else str(None)]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns params with non-primitvies converted to strings for logging.
        """
        return {k: v if type(v) in [bool, int, float, str, torch.Tensor] else str(v) for k, v in params.items()}

    def log_best_epoch_to_json(self, best_epoch_metrics: Dict[str, float]):
        with open(os.path.join(self.log_dir, "metrics_best_epoch.json"), "w") as filepath:
            json.dump(
                obj=best_epoch_metrics,
                fp=filepath
            )

    def add_preds_trgs(self, batch_output: Dict):
        if len(self.trgs[0]) < self.len_logged_preds_trgs:
            if len(self.trgs[0]) == 0 or self.trgs[0][0] != batch_output['targets_decoded'][0][0]:
                for i, trg in enumerate(batch_output['targets_decoded']):
                    if len(self.trgs) <= i:
                        self.trgs.append(trg)
                    else:
                        self.trgs[i].extend(trg)
        if len(self.current_preds) < self.len_logged_preds_trgs:
            self.current_preds.extend(batch_output['predictions_decoded'])

    def reset_logged_preds(self):
        self.current_preds = []

    def set_current_preds_to_best(self):
        self.best_preds = self.current_preds

    def log_best_preds_and_trgs_to_txt(self):
        with open(os.path.join(self.log_dir, "best_preds.txt"), "w") as text_file:
            text_file.write("\n".join(self.best_preds))
        for i, trgs in enumerate(self.trgs):
            with open(os.path.join(self.log_dir, f"trgs{i}.txt"), "w") as text_file:
                text_file.write("\n".join(trgs))

    @staticmethod
    def _metrics_to_str(metrics: Dict[str, Any]) -> str:
        log_message = '\n'
        max_name_length = max(len(name) for name in metrics.keys())

        for name, value in metrics.items():
            if "relevant" not in name:
                if 'clf_report' in name:
                    value = TrainLogger._clf_report_dict_to_str(clf_report=value)
                if isinstance(value, numbers.Number):
                    value = round(value, 4)
                log_message += f'{name:{max_name_length}} {value}\n'
        return log_message

    @staticmethod
    def _clf_report_dict_to_str(clf_report: Dict) -> str:
        if clf_report["micro avg"].get("strict", False):
            metric_types = ["strict", "partial_type", "exact", "partial"]
            report = ""
            for metric_type in metric_types:
                headers = ('Type', 'Precision', 'Recall', 'F1', 'Support')
                digits = 2
                max_name_width = max(len(name) for name in clf_report.keys())
                head_fmt = '    {:>{max_name_width}s} ' + ' {:>9}' * (len(headers) - 1)
                report += f"    \n {metric_type} \n"
                report += head_fmt.format(*headers, max_name_width=max_name_width)
                report += '    \n\n'
                row_fmt = '    {:>{max_name_width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
                for name, metric_dict in clf_report.items():
                    if name == 'micro avg':
                        report += '    \n'
                    metrics = metric_dict[metric_type]
                    row = (name,) + tuple(metrics.values())
                    report += row_fmt.format(*row, max_name_width=max_name_width, digits=digits)
                report += "\n"
        else:
            headers = ('Type', 'Precision', 'Recall', 'F1', 'Support')
            digits = 2
            max_name_width = max(len(name) for name in clf_report.keys())
            head_fmt = '    {:>{max_name_width}s} ' + ' {:>9}' * (len(headers) - 1)
            report = '    \n'
            report += head_fmt.format(*headers, max_name_width=max_name_width)
            report += '    \n\n'
            row_fmt = '    {:>{max_name_width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
            for name, metrics in clf_report.items():
                if name == 'micro avg':
                    report += '    \n'
                row = (name,) + tuple(metrics.values())
                report += row_fmt.format(*row, max_name_width=max_name_width, digits=digits)
        return report

    @classmethod
    def from_config(cls,
                    type_: str,
                    *args,
                    **kwargs):
        try:
            callable_path = TRAIN_LOGGERS[type_]
            parts = callable_path.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'{cls.__name__} "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)

        return class_(*args, **kwargs)
