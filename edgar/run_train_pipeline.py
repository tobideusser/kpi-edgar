import argparse
import datetime
import logging
import os

import yaml
from fluidml import Flow, Swarm
from fluidml.flow import TaskSpec

from edgar import project_path
from edgar.tasks import DataParsing
from edgar.utils.fluid_helper import configure_logging, MyLocalFileStore, TaskResource
from edgar.utils.training_utils import get_balanced_devices

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default=os.path.join(project_path, 'configs', 'kpi_relation_extraction', 'config.yaml'),
                        type=str,
                        help="Path to config")
    parser.add_argument('--cuda-ids',
                        default=None,
                        type=int,
                        nargs='+',
                        help="GPU ids, e.g. `--cuda-ids 0 1`")
    parser.add_argument('--use-cuda',
                        action='store_true',
                        help="Use cuda.")
    parser.add_argument('--warm-start',
                        action='store_true',
                        help="Tries to warm start training.")
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help="Number of multiprocessing workers.")
    parser.add_argument('--force',
                        type=str,
                        nargs='+',
                        default=None,
                        help="Task or tasks to force execute. '+' registers successor tasks also for force execution."
                             "E.g. --force ModelTraining+")
    parser.add_argument('--gs-expansion-method',
                        type=str,
                        default='product',
                        choices=['product', 'zip'],
                        help="Method to expand config for grid search")
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
    warm_start = args.warm_start  # False  # continue training from an existing checkpoint
    gs_expansion_method: str = args.gs_expansion_method

    log_dir = os.path.join(base_dir, "logging")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(level="INFO", log_dir=log_dir)

    # get task configs
    data_parsing_cfg = config["DataParsing"]

    # create all task specs
    data_parsing = TaskSpec(task=DataParsing, config=data_parsing_cfg)

    # dependencies between tasks

    # all tasks
    tasks = [
        data_parsing
    ]

    # create list of resources
    devices = get_balanced_devices(count=num_workers, use_cuda=use_cuda, cuda_ids=cuda_ids)
    resources = [TaskResource(device=devices[i]) for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = MyLocalFileStore(base_dir=base_dir)

    start = datetime.datetime.now()
    with Swarm(n_dolphins=num_workers,
               resources=resources,
               results_store=results_store) as swarm:
        flow = Flow(swarm=swarm)
        flow.create(task_specs=tasks)

        # visualize expanded task graph
        # visualize_graph_in_console(flow.task_graph, use_unicode=False, use_pager=True)

        flow.run(force=force)
    end = datetime.datetime.now()
    logger.info(f'{end - start}')


if __name__ == "__main__":
    main()
