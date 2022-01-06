import logging
from typing import Optional, List

from fluidml import Task

from edgar.data_parsing import EdgarParser


logger = logging.getLogger(__name__)


class DataParsing(Task):

    publishes = ['corpus_parsed']

    def __init__(
            self,
            data_folder: str,
            entity_prefixes: List[str],
            entity_formats: List[str],
            path_to_data_folders: str,
            dataset_name: Optional[str] = "EDGAR",
            debug_size: Optional[int] = None,
            train_mode: bool = True
    ):
        super().__init__()

        # config params
        self.dataset_name = dataset_name
        self.debug_size = debug_size
        self.data_folder = data_folder
        self.entity_prefixes = entity_prefixes
        self.entity_formats = entity_formats
        self.path_to_data_folders = path_to_data_folders

        self.train_mode = train_mode

    def run(self):
        logger.info(f'Parse {self.dataset_name} dataset with debug size {self.debug_size}.')
        parser = EdgarParser(
            entity_prefixes=self.entity_prefixes,
            entity_formats=self.entity_formats,
            path_to_data_folders=self.path_to_data_folders,
            debug_size=self.debug_size
        )
        corpus = parser.parse_data_folder()

        # if self.train_mode:
        #     self.save(corpus.to_dict(), 'corpus_parsed', type_='pickle')
        # else:
        #     return corpus
