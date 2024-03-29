import argparse
import datetime
import logging
import os

import yaml
from fluidml import Flow
from fluidml.flow import TaskSpec, GridTaskSpec

from edgar import project_path
from edgar.tasks import (
    DataParsing,
    DataTokenizing,
    DataTagging,
    AnnotationMerging,
    SubWordTokenization,
    ModelTraining,
    ModelPredicting,
)
from edgar.utils.fluid_helper import configure_logging, MyLocalFileStore, TaskResource
from edgar.utils.training_utils import get_balanced_devices

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(project_path, "configs", "config.yaml"),
        type=str,
        help="Path to config",
    )
    parser.add_argument(
        "--cuda-ids",
        default=None,
        type=int,
        nargs="+",
        help="GPU ids, e.g. `--cuda-ids 0 1`",
    )
    parser.add_argument("--use-cuda", action="store_true", help="Use cuda.")
    parser.add_argument(
        "--warm-start", action="store_true", help="Tries to warm start training."
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of multiprocessing workers."
    )
    parser.add_argument(
        "--force",
        type=str,
        nargs="+",
        default=None,
        help="Task or tasks to force execute. '+' registers successor tasks also for force execution."
        "E.g. --force ModelTraining+",
    )
    parser.add_argument(
        "--gs-expansion-method",
        type=str,
        default="product",
        choices=["product", "zip"],
        help="Method to expand config for grid search",
    )
    parser.add_argument(
        "--log-to-tmux", action="store_true", help="Log to several tmux panes."
    )
    parser.add_argument(
        "--project-name", type=str, default="edgar", help="Name of project."
    )
    parser.add_argument("--run-name", type=str, default=None, help="Name of run.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))

    base_dir = config["base_dir"]

    # Parse run settings from argparse (defaults and choices see above in argparse)
    num_workers = args.num_workers  # 1
    force = args.force  # 'ModelTraining+'
    use_cuda = args.use_cuda
    cuda_ids = args.cuda_ids  # [1]  # [0, 1]
    warm_start = (
        args.warm_start
    )  # False  # continue training from an existing checkpoint
    gs_expansion_method: str = args.gs_expansion_method

    log_dir = os.path.join(base_dir, "logging")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(level="INFO", log_dir=log_dir)

    # get task configs
    data_parsing_cfg = config["DataParsing"]
    data_tokenization_cfg = config["DataTokenizing"]
    data_tagging_cfg = config["DataTagging"]
    annotation_merging_cfg = config["AnnotationMerging"]
    sub_word_tokenization_cfg = config["SubWordTokenization"]
    model_training_cfg = config["ModelTraining"]
    model_training_additional_kwargs = {
        "checkpointer_params": {
            "serialization_dir": "models",
            "num_serialized_models_to_keep": 1,
        },
        "train_logger_params": {
            "type_": "tensorboard",
            "log_dir": "logs",
        },
        "warm_start": warm_start,
    }

    # create all task specs
    data_parsing = TaskSpec(task=DataParsing, config=data_parsing_cfg)
    data_tokenizing = TaskSpec(task=DataTokenizing, config=data_tokenization_cfg)
    data_tagging = TaskSpec(task=DataTagging, config=data_tagging_cfg)
    annotation_merging = TaskSpec(task=AnnotationMerging, config=annotation_merging_cfg)
    sub_word_tokenization = GridTaskSpec(
        task=SubWordTokenization, gs_config=sub_word_tokenization_cfg
    )
    model_training = GridTaskSpec(
        task=ModelTraining,
        gs_config=model_training_cfg,
        gs_expansion_method=gs_expansion_method,
        additional_kwargs=model_training_additional_kwargs,
    )
    model_predicting = TaskSpec(task=ModelPredicting, config=config["ModelPredicting"])

    # dependencies between tasks
    data_tokenizing.requires(data_parsing)
    data_tagging.requires(data_tokenizing)
    annotation_merging.requires(data_tagging)
    sub_word_tokenization.requires(annotation_merging)
    model_training.requires([sub_word_tokenization, annotation_merging])
    model_predicting.requires(
        [model_training, sub_word_tokenization, data_tagging, annotation_merging]
    )
    # all tasks
    tasks = [
        data_parsing,
        data_tokenizing,
        data_tagging,
        annotation_merging,
        sub_word_tokenization,
        model_training,
        model_predicting,
    ]

    # create list of resources
    devices = get_balanced_devices(
        count=num_workers, use_cuda=use_cuda, cuda_ids=cuda_ids
    )
    resources = [TaskResource(device=devices[i]) for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = MyLocalFileStore(base_dir=base_dir)

    start = datetime.datetime.now()

    # create flow (expanded task graph)
    flow = Flow(tasks=tasks)
    # run linearly without swarm if num_workers is set to 1
    # note resources are now assigned equally to all tasks (e.g. device info)
    # else run graph in parallel using multiprocessing
    # create list of resources which is distributed among workers
    # e.g. to manage that each worker has dedicated access to specific gpus
    flow.run(
        num_workers=args.num_workers,
        resources=resources,
        log_to_tmux=args.log_to_tmux,
        force=args.force,
        results_store=results_store,
        project_name=args.project_name,
        run_name=args.run_name,
    )

    end = datetime.datetime.now()
    logger.info(f"{end - start}")


if __name__ == "__main__":
    main()
