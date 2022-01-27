import logging
import os
from typing import List, Tuple, Optional, Dict, Any

from fluidml.common import Task
import torch

from edgar.trainer.utils import get_device

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self,
                 task: Task,
                 serialization_dir: str = 'models',
                 num_serialized_models_to_keep: int = 2):
        self._task = task
        run_dir = self._task.get_store_context()
        self._serialization_dir = serialization_dir
        self._model_dir = os.path.join(run_dir, serialization_dir)
        os.makedirs(self._model_dir, exist_ok=True)
        self._serialized_paths: List[Tuple[str, str]] = []
        self._num_serialized_models_to_keep = num_serialized_models_to_keep

    def save_checkpoint(
            self,
            model_state: Dict,
            training_state: Dict,
            epoch: int,
            is_best_so_far: bool = False
    ) -> None:
        epoch = str(epoch).zfill(2)
        self._task.save(obj=model_state, name=f"model_state_epoch_{epoch}", type_='torch',
                        sub_dir=self._serialization_dir)
        self._task.save(obj=training_state, name=f"training_state_epoch_{epoch}", type_='torch',
                        sub_dir=self._serialization_dir)

        if is_best_so_far:
            logger.info(f"Best validation performance so far. Overwriting '{self._model_dir}/best_model.pt'.")
            self._task.save(obj=model_state, name=f"best_model", type_='torch',
                            sub_dir=self._serialization_dir)

        if self._num_serialized_models_to_keep is not None and self._num_serialized_models_to_keep >= 0:
            model_state_name = f"model_state_epoch_{epoch}"
            training_state_name = f"training_state_epoch_{epoch}"
            self._serialized_paths.append((model_state_name, training_state_name))
            if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                objects_to_remove = self._serialized_paths.pop(0)
                for obj_name in objects_to_remove:
                    self._task.delete(obj_name)

    def load_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes a training state (typically consisting of an epoch count and optimizer state),
        which is serialized separately from  model parameters. This function should only be used to
        continue training - if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`
        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return empty dicts.
        # Returns
        states : `Tuple[Dict[str, Any], Dict[str, Any]]`
            The model state and the training state.
        """
        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return {}, {}

        model_state_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_state_path, map_location=get_device())
        training_state = torch.load(training_state_path, map_location=get_device())
        return model_state, training_state

    def _find_latest_checkpoint(self) -> Optional[Tuple[str, str]]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        checkpoint_exists = any("model_state_epoch_" in x for x in os.listdir(self._model_dir))

        if not checkpoint_exists:
            return None

        serialization_files = os.listdir(self._model_dir)
        epochs = [int(path.split('.pt')[0].split('_')[-1])
                  for path in serialization_files if "model_state_epoch" in path]

        last_epoch = sorted(epochs, reverse=True)[0]
        last_epoch = str(last_epoch).zfill(2)

        model_state_path = os.path.join(self._model_dir, f"model_state_epoch_{last_epoch}.pt")
        training_state_path = os.path.join(self._model_dir, f"training_state_epoch_{last_epoch}.pt")

        return model_state_path, training_state_path
