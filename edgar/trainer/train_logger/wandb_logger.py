from argparse import Namespace
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import wandb
from wandb.wandb_run import Run

from edgar.trainer.train_logger import TrainLogger


logger = logging.getLogger(__name__)


class WandbLogger(TrainLogger):
    r"""
    Log using `Weights and Biases <https://www.wandb.com/>`_.
    Install it with pip:
    .. code-block:: bash
        pip install wandb
    Args:
        name: Display name for the run.
        save_dir: Path where data is saved.
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        anonymous: Enables or explicitly disables anonymous logging.
        version: Sets the version, mainly used to resume a previous run.
        project: The name of the project to which this run will belong.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        experiment: WandB experiment object.
        prefix: A string to put at the beginning of metric keys.
        \**kwargs: Additional arguments like `entity`, `group`, `tags`, etc. used by
            :func:`wandb.init` can be passed as keyword arguments in this logger.
    Example::
    .. code-block:: python
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer
        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)
    Note: When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
    make sure to use `commit=False` so the logging step does not increase.
    See Also:
        - `Tutorial <https://app.wandb.ai/cayush/pytorchlightning/reports/
          Use-Pytorch-Lightning-with-Weights-%26-Biases--Vmlldzo2NjQ1Mw>`__
          on how to use W&B with Pytorch Lightning.
    """

    def __init__(
            self,
            log_dir: str,
            name: Optional[str] = None,
            offline: bool = False,
            id_: Optional[str] = None,
            anonymous: bool = False,
            version: Optional[str] = None,
            project: Optional[str] = None,
            log_model: bool = False,
            experiment=None,
            prefix: str = '',
            **kwargs
    ):
        if wandb is None:
            raise ImportError('You want to use `wandb` logger which is not installed yet,'  # pragma: no-cover
                              ' install it with `pip install wandb`.')
        super().__init__(log_dir=log_dir)
        self._name = name
        self._anonymous = 'allow' if anonymous else None
        self._id = version or id_
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._kwargs = kwargs
        # logging multiple Trainer on a single W&B run (k-fold, resuming, etc)
        self._step_offset = 0

        if offline:
            os.environ['WANDB_MODE'] = 'dryrun'

        self._writer: Run = wandb.init(name=self._name, dir=self.log_dir, project=self._project,
                                       anonymous=self._anonymous, id=self._id, resume='allow', **self._kwargs)
        # offset logging step when resuming a run
        self._step_offset = self._writer.step

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        self._writer.watch(model, log=log, log_freq=log_freq)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)
        self._writer.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:

        if step is not None and step + self._step_offset < self._writer.step:
            logger.warning('Trying to log at a previous step. Use `commit=False` when logging metrics manually.')
        self._writer.log(metrics, step=(step + self._step_offset) if step is not None else None)

    def log_graph(self, model: nn.Module, input_array: Optional[torch.Tensor] = None) -> None:
        pass

    @property
    def name(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._writer.project_name() if self._writer else self._name

    def finalize(self, status: str) -> None:
        # offset future training logged on same W&B run
        if self._writer is not None:
            self._step_offset = self._writer.step

        # upload all checkpoints from saving dir
        if self._log_model:
            self._writer.save(os.path.join(self.log_dir, "*.ckpt"))


"""
Arguments:
project: (str, optional) The name of the project where you're sending
    the new run. If the project is not specified, the run is put in an
    "Uncategorized" project.
entity: (str, optional) An entity is a username or team name where
    you're sending runs. This entity must exist before you can send runs
    there, so make sure to create your account or team in the UI before
    starting to log runs.
    If you don't specify an entity, the run will be sent to your default
    entity, which is usually your username. Change your default entity
    in [Settings](wandb.ai/settings) under "default location to create
    new projects".
config: (dict, argparse, absl.flags, str, optional)
    This sets wandb.config, a dictionary-like object for saving inputs
    to your job, like hyperparameters for a model or settings for a data
    preprocessing job. The config will show up in a table in the UI that
    you can use to group, filter, and sort runs. Keys should not contain
    `.` in their names, and values should be under 10 MB.
    If dict, argparse or absl.flags: will load the key value pairs into
    the wandb.config object.
    If str: will look for a yaml file by that name, and load config from
    that file into the wandb.config object.
save_code: (bool, optional) Turn this on to save the main script or
    notebook to W&B. This is valuable for improving experiment
    reproducibility and to diff code across experiments in the UI. By
    default this is off, but you can flip the default behavior to "on"
    in [Settings](wandb.ai/settings).
group: (str, optional) Specify a group to organize individual runs into
    a larger experiment. For example, you might be doing cross
    validation, or you might have multiple jobs that train and evaluate
    a model against different test sets. Group gives you a way to
    organize runs together into a larger whole, and you can toggle this
    on and off in the UI. For more details, see
    [Grouping](docs.wandb.com/library/grouping).
job_type: (str, optional) Specify the type of run, which is useful when
    you're grouping runs together into larger experiments using group.
    For example, you might have multiple jobs in a group, with job types
    like train and eval. Setting this makes it easy to filter and group
    similar runs together in the UI so you can compare apples to apples.
tags: (list, optional) A list of strings, which will populate the list
    of tags on this run in the UI. Tags are useful for organizing runs
    together, or applying temporary labels like "baseline" or
    "production". It's easy to add and remove tags in the UI, or filter
    down to just runs with a specific tag.
name: (str, optional) A short display name for this run, which is how
    you'll identify this run in the UI. By default we generate a random
    two-word name that lets you easily cross-reference runs from the
    table to charts. Keeping these run names short makes the chart
    legends and tables easier to read. If you're looking for a place to
    save your hyperparameters, we recommend saving those in config.
notes: (str, optional) A longer description of the run, like a -m commit
    message in git. This helps you remember what you were doing when you
    ran this run.
dir: (str, optional) An absolute path to a directory where metadata will
    be stored. When you call download() on an artifact, this is the
    directory where downloaded files will be saved. By default this is
    the ./wandb directory.
sync_tensorboard: (bool, optional) Whether to copy all TensorBoard logs
    to W&B (default: False).
    [Tensorboard](https://docs.wandb.com/integrations/tensorboard)
resume (bool, str, optional): Sets the resuming behavior. Options:
    "allow", "must", "never", "auto" or None. Defaults to None.
    Cases:
    - None (default): If the new run has the same ID as a previous run,
    this run overwrites that data.
    - "auto" (or True): if the preivous run on this machine crashed,
    automatically resume it. Otherwise, start a new run.
    - "allow": if id is set with init(id="UNIQUE_ID") or
    WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run,
    wandb will automatically resume the run with that id. Otherwise,
    wandb will start a new run.
    - "never": if id is set with init(id="UNIQUE_ID") or
    WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run,
    wandb will crash.
    - "must": if id is set with init(id="UNIQUE_ID") or
    WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run,
    wandb will automatically resume the run with the id. Otherwise
    wandb will crash.
    See https://docs.wandb.com/library/advanced/resuming for more.
reinit: (bool, optional) Allow multiple wandb.init() calls in the same
    process. (default: False)
magic: (bool, dict, or str, optional) The bool controls whether we try to
    auto-instrument your script, capturing basic details of your run
    without you having to add more wandb code. (default: False)
    You can also pass a dict, json string, or yaml filename.
config_exclude_keys: (list, optional) string keys to exclude from
    `wandb.config`.
config_include_keys: (list, optional) string keys to include in
    wandb.config.
anonymous: (str, optional) Controls anonymous data logging. Options:
    - "never" (default): requires you to link your W&B account before
    tracking the run so you don't accidentally create an anonymous
    run.
    - "allow": lets a logged-in user track runs with their account, but
    lets someone who is running the script without a W&B account see
    the charts in the UI.
    - "must": sends the run to an anonymous account instead of to a
    signed-up user account.
mode: (str, optional) Can be "online", "offline" or "disabled". Defaults to
    online.
allow_val_change: (bool, optional) Whether to allow config values to
    change after setting the keys once. By default we throw an exception
    if a config value is overwritten. If you want to track something
    like a varying learning_rate at multiple times during training, use
    wandb.log() instead. (default: False in scripts, True in Jupyter)
force: (bool, optional) If True, this crashes the script if a user isn't
    logged in to W&B. If False, this will let the script run in offline
    mode if a user isn't logged in to W&B. (default: False)
sync_tensorboard: (bool, optional) Synchronize wandb logs from tensorboard or
    tensorboardX and saves the relevant events file. Defaults to false.
monitor_gym: (bool, optional) automatically logs videos of environment when
    using OpenAI Gym. (default: False)
    See https://docs.wandb.com/library/integrations/openai-gym
id: (str, optional) A unique ID for this run, used for Resuming. It must
    be unique in the project, and if you delete a run you can't reuse
    the ID. Use the name field for a short descriptive name, or config
    for saving hyperparameters to compare across runs. The ID cannot
    contain special characters.
    See https://docs.wandb.com/library/resuming
"""