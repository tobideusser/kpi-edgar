import logging
from typing import Dict, Union

from fluidml import Task

from edgar.data_classes import Corpus
from edgar.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class DataTokenizing(Task):

    publishes = ["corpus_tokenized"]

    def __init__(self, remove_tables: bool = False, language: str = "en", train_mode: bool = True):
        super().__init__()

        # config params
        self.remove_tables = remove_tables
        self.language = language

        self.train_mode = train_mode

    def run(self, corpus_parsed: Union[Dict, Corpus]):

        if isinstance(corpus_parsed, Dict):
            logger.info("Converting corpus_parsed dict to Corpus object.")
            corpus = Corpus.from_dict(corpus_parsed)
        else:
            corpus = corpus_parsed

        logger.info("Tokenize corpus...")
        tokenizer = Tokenizer(remove_tables=self.remove_tables, language=self.language)
        tokenizer.tokenize_corpus(corpus=corpus)

        if self.train_mode:
            self.save(corpus.to_dict(), "corpus_tokenized", type_="pickle")
        else:
            return corpus
