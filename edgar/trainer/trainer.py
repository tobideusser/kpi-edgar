import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from edgar.trainer import Checkpointer, Evaluator, MetricTracker, LearningRateScheduler, TrainLogger
from edgar.trainer.utils import clamp_tensor, set_seeds, detach_batch

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        evaluator: Optional[Evaluator] = None,
        valid_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LearningRateScheduler] = None,
        checkpointer: Optional[Checkpointer] = None,
        train_logger: Optional[TrainLogger] = None,  # e.g. WandBLogger or TensorboardLogger
        unique_config: Optional[Dict] = None,
        num_epochs: int = 10,
        valid_metric: str = "-loss",
        early_stopping_patience: Optional[int] = None,
        num_grad_accumulation_steps: int = 1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._evaluator = evaluator
        self._lr_scheduler = lr_scheduler
        self._checkpointer = checkpointer
        self._train_logger = train_logger
        self._unique_config = unique_config

        self._num_epochs = num_epochs
        self._num_grad_accumulation_steps = num_grad_accumulation_steps
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self.metric_tracker = MetricTracker(patience=early_stopping_patience, metric_name=valid_metric)
        self._valid_metric = valid_metric[1:]  # Get rid of + or -

        self.current_epoch = None

    def _valid_batch(self, batch) -> Dict:
        output = self._model.predict(batch)
        output["loss"] = output["loss"].item()
        output = detach_batch(output)
        torch.cuda.empty_cache()
        return output

    def _train_batch(self, batch, step: int) -> Dict:
        output = self._model(batch)
        loss = output["loss"]
        loss = loss / self._num_grad_accumulation_steps
        loss.backward()
        output["loss"] = loss.item()
        if step % self._num_grad_accumulation_steps == 0:
            # scales gradiens if grad_norm is set
            # TODO: returned batch_grad_norm could be included in model logging
            self._rescale_gradients()
            self._optimizer.step()
            if self._lr_scheduler and self._lr_scheduler.interval == "step":
                self._lr_scheduler.step()
            self._optimizer.zero_grad()
        output = detach_batch(output)
        torch.cuda.empty_cache()
        return output

    def _train_epoch(self) -> Dict[str, Any]:
        train_loss = 0.0
        train_metrics: Dict[str, Any] = {}
        self._model.train()

        set_seeds()
        for step, batch in enumerate(tqdm(self._train_dataloader), 1):
            batch_output: Dict = self._train_batch(batch=batch, step=step)
            loss = batch_output["loss"]
            train_loss += loss

            if self._evaluator:
                self._evaluator.increment_metrics(batch_output=batch_output)

        if self._evaluator:
            train_metrics: Dict[str, Any] = self._evaluator.get_metrics(reset=True)
        num_train_batches = len(self._train_dataloader) if len(self._train_dataloader) > 0 else 1
        train_metrics["loss"] = train_loss / num_train_batches

        return train_metrics

    def _validate_epoch(self) -> Dict:
        valid_loss = 0.0
        valid_metrics: Dict[str, Any] = {}
        self._model.eval()

        set_seeds()
        for batch in tqdm(self._valid_dataloader):
            batch_output: Dict = self._valid_batch(batch=batch)
            loss = batch_output.get("loss")
            if loss is not None:
                valid_loss += loss
            if self._evaluator:
                self._evaluator.increment_metrics(batch_output=batch_output)

        if self._evaluator:
            valid_metrics: Dict[str, Any] = self._evaluator.get_metrics(reset=True)
        num_valid_batches = len(self._valid_dataloader) if len(self._valid_dataloader) > 0 else 1
        valid_metrics["loss"] = valid_loss / num_valid_batches

        return valid_metrics

    def train(self, warm_start: bool = True):

        logger.info("Starting training.")

        valid_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = 0.0
        metrics: Dict[str, Any] = {}

        self.current_epoch = self._restore_checkpoint(warm_start=warm_start)

        self._enable_gradient_clipping()

        while self.current_epoch <= self._num_epochs:
            logger.info(f"Epoch {self.current_epoch}:\n")

            train_metrics = self._train_epoch()
            if self._valid_dataloader is not None:
                with torch.no_grad():
                    valid_metrics = self._validate_epoch()

                    # Check validation metric for early stopping
                    this_epoch_val_metric = valid_metrics[self._valid_metric]
                    self.metric_tracker.add_metric(this_epoch_val_metric)

                    if self.metric_tracker.should_stop_early():
                        logger.info("Ran out of patience. Stopping training.")
                        break

            # Create overall metrics dict
            metrics["epoch"] = self.current_epoch

            if "relevant_re_clf_report" in train_metrics:
                for name, value in train_metrics["relevant_re_clf_report"].items():
                    metrics["training_" + name] = value
            else:
                for name, value in train_metrics.items():
                    metrics["training_" + name] = value

            if "relevant_re_clf_report" in valid_metrics:
                for name, value in valid_metrics["relevant_re_clf_report"].items():
                    metrics["validation_" + name] = value
            else:
                for name, value in valid_metrics.items():
                    metrics["validation_" + name] = value

            if self.metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = self.current_epoch
                for key, value in valid_metrics.items():
                    metrics["best_validation_" + key] = value

                self.metric_tracker.best_epoch_metrics = valid_metrics

            if self._train_logger:
                if self.current_epoch == 1:
                    self._train_logger.log_hyperparams(
                        params=self._unique_config,
                        metrics={f"best_validation_{k}": v for k, v in valid_metrics.items()},
                    )
                self._train_logger.log_metrics(metrics=metrics, step=self.current_epoch)
                # TODO: Convert metrics to clf report
                # self._train_logger.log_clf_report(metrics=metrics, step=self.current_epoch)

            if self._lr_scheduler and self._lr_scheduler.interval == "epoch":
                self._lr_scheduler.step(metric=this_epoch_val_metric)

            if self._checkpointer is not None:
                model_state, training_state = self._get_checkpoint_state()
                self._checkpointer.save_checkpoint(
                    model_state=model_state,
                    training_state=training_state,
                    epoch=self.current_epoch,
                    is_best_so_far=self.metric_tracker.is_best_so_far(),
                )
            self.current_epoch += 1

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda grad: clamp_tensor(grad, minimum=-self._grad_clipping, maximum=self._grad_clipping)
                    )

    def _rescale_gradients(self):  # -> torch.Tensor:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        Returns the norm of the gradients.
        """
        # parameters_to_clip = [p for p in self._model.parameters() if p.grad is not None]
        if self._grad_norm:
            parameters_to_clip = [p for p in self._model.parameters() if p.grad is not None]
            clip_grad_norm_(parameters_to_clip, self._grad_norm)
        #     return clip_grad_norm_(parameters_to_clip, self._grad_norm)
        # else:
        #     return torch.norm(
        #         torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
        #     )

    def _get_checkpoint_state(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        model_state = self._model.state_dict()

        # These are the training states we need to persist.
        training_state = {
            "metric_tracker": self.metric_tracker.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "epoch": self.current_epoch,
        }

        # If we have a learning rate, we should persist it too.
        if self._lr_scheduler is not None:
            training_state["lr_scheduler"] = self._lr_scheduler.state_dict()

        return model_state, training_state

    def _restore_checkpoint(self, warm_start: bool = True) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load('/path/to/model/weights.th'))`
        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.
        # Returns
        epoch: `int`
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        if self._checkpointer is None or not warm_start:
            return 1

        model_state, training_state = self._checkpointer.load_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 1

        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        if self._lr_scheduler is not None and "lr_scheduler" in training_state:
            self._lr_scheduler.load_state_dict(training_state["lr_scheduler"])

        # TODO: Move optimizer to cuda?
        # training_util.move_optimizer_to_cuda(self.optimizer)

        if "metric_tracker" in training_state:
            self.metric_tracker.load_state_dict(training_state["metric_tracker"])

        epoch_to_return = training_state["epoch"] + 1

        return epoch_to_return


# def __init__(
#         self,
#         model: Model,
#         optimizer: torch.optim.Optimizer,
#         data_loader: DataLoader,
#         patience: Optional[int] = None,
#         validation_metric: str = '-loss',
#         validation_data_loader: DataLoader = None,
#         num_epochs: int = 20,
#         serialization_dir: Optional[str] = None,
#         checkpointer: Checkpointer = None,
#         cuda_device: Optional[Union[int, torch.device]] = None,
#         grad_norm: Optional[float] = None,
#         grad_clipping: Optional[float] = None,
#         learning_rate_scheduler: Optional[LearningRateScheduler] = None,
#         momentum_scheduler: Optional[MomentumScheduler] = None,
#         tensorboard_writer: TensorboardWriter = None,
#         moving_average: Optional[MovingAverage] = None,
#         batch_callbacks: List[BatchCallback] = None,
#         epoch_callbacks: List[EpochCallback] = None,
#         end_callbacks: List[EpochCallback] = None,
#         trainer_callbacks: List[TrainerCallback] = None,
#         distributed: bool = False,
#         local_rank: int = 0,
#         world_size: int = 1,
#         num_gradient_accumulation_steps: int = 1,
#         use_amp: bool = False,
