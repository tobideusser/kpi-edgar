from argparse import Namespace
import logging
from typing import Any, Dict, Optional, Union, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from edgar.trainer.train_logger import TrainLogger


logger = logging.getLogger(__name__)


class TensorBoardLogger(TrainLogger):
    r"""
    Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format.
    Implemented using :class:`~torch.utils.tensorboard.SummaryWriter`. Logs are saved to
    ``os.path.join(save_dir)``. This is the default logger in Lightning, it comes
    preinstalled.
    Args:
        save_dir: Save directory
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        \**kwargs: Additional arguments like `comment`, `filename_suffix`, etc. used by
            :class:`SummaryWriter` can be passed as keyword arguments in this logger.
    """
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(
            self,
            log_dir: str,
            log_graph: bool = True,
            default_hp_metric: bool = True,
            len_logged_preds_trgs: Optional[Union[int, str]] = None,
            **kwargs
    ):
        super().__init__(log_dir=log_dir, len_logged_preds_trgs=len_logged_preds_trgs)
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        # self._fs = get_filesystem(save_dir)

        self._writer = SummaryWriter(log_dir=self.log_dir, **kwargs)

        self.hparams = {}
        self._kwargs = kwargs

    def log_samples(
            self,
            batch_output: Dict[str, Any],
            metrics: Dict[str, Any],
            src_tables: List[str],
            prefix: str, step:
            Optional[int] = None
    ):
        targets = "".join([str(i) + ". " + trg + "\n" for i, trg in enumerate(batch_output['targets_decoded'][0])])
        predictions = "".join(
            [str(i) + ". " + pred + "\n" for i, pred in enumerate(batch_output['predictions_decoded'])]
        )
        tables = "".join([str(i) + ". `" + src_table + "`\n" for i, src_table in enumerate(src_tables)])

        markdown_summary = "### Targets\n" + targets + "### Predictions\n" + predictions + "### Table Source\n" + tables
        self._writer.add_text(
            tag=prefix + " Text Samples",
            text_string=markdown_summary,
            global_step=step
        )

        text_samples = {
            "prd_sample": batch_output['predictions_decoded'][0],
            "trg_sample": batch_output['targets_decoded'][0][0]
        }

        log_message = TrainLogger._metrics_to_str(metrics=metrics)
        logger.info(log_message)

    def log_hyperparams(self,
                        params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        params = self._convert_params(params)

        # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)
            exp, ssi, sei = hparams(params, metrics)
            self._writer.file_writer.add_summary(exp)  # self._get_file_writer() instead of file_writer
            self._writer.file_writer.add_summary(ssi)
            self._writer.file_writer.add_summary(sei)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if 'clf_report' in k:
                v = TrainLogger._clf_report_dict_to_str(clf_report=v)

            if isinstance(v, dict):
                self._writer.add_scalars(k, v, step)
            elif isinstance(v, str):
                self._writer.add_text(k, v, step)
            else:
                try:
                    self._writer.add_scalar(k, v, step)
                except Exception as e:
                    logger.warning(f'You tried to log {v} which is not currently supported. '
                                   f'Try a dict or a scalar/tensor.')
                    logger.exception(e)
        logger.info(TrainLogger._metrics_to_str(metrics))

    def log_graph(self, model: nn.Module, input_array: Optional[torch.Tensor] = None):
        if self._log_graph:
            input_array = model.transfer_batch_to_device(input_array, model.device)
            self._writer.add_graph(model, input_array)

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        pass

    def finalize(self) -> None:
        self._writer.flush()
        self._writer.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
