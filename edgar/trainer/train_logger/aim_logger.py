import logging
from typing import Any, Dict, Optional, Union, List

import aim
import torch
import torch.nn as nn

from edgar.trainer.train_logger import TrainLogger


logger = logging.getLogger(__name__)


class AimBoardLogger(TrainLogger):

    def __init__(
            self,
            log_dir: Optional[str] = None,
            experiment: Optional[str] = None,
            run: Optional[str] = None,
            len_logged_preds_trgs: Optional[Union[int, str]] = None,
            **kwargs
    ):
        super().__init__(log_dir=log_dir, len_logged_preds_trgs=len_logged_preds_trgs)

        self._session = aim.Session(repo=log_dir,
                                    experiment=experiment,
                                    run=run,
                                    **kwargs)

    def log_samples(
            self,
            batch_output: Dict[str, Any],
            metrics: Dict[str, Any],
            src_tables: List[str],
            prefix: str, step:
            Optional[int] = None
    ):
        pass

    def log_hyperparams(self,
                        params: Dict[str, Any],
                        metrics=None) -> None:
        self._session.set_params(params, 'hyper parameters')

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if 'clf_report' in k:
                v = TrainLogger._clf_report_dict_to_str(clf_report=v)

            if isinstance(v, Dict):
                self.log_metrics(v, step)
            else:
                self._session.track(v, name=k, epoch=step)
        logger.info(TrainLogger._metrics_to_str(metrics))

    def log_graph(self, model: nn.Module, input_array: Optional[torch.Tensor] = None):
        pass

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        pass

    def finalize(self) -> None:
        self._session.flush()
        self._session.close()
