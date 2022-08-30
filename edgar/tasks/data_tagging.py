import logging
from typing import Dict, Union

from fluidml import Task
from tqdm import tqdm

from edgar.currency_tagger import tagger
from edgar.data_classes import Corpus, Paragraph, Table


logger = logging.getLogger(__name__)


class DataTagging(Task):

    publishes = ["corpus_tagged"]

    def __init__(self, remove_non_currency_sentences: bool = True, language: str = "en", train_mode: bool = True):
        super().__init__()

        # config params
        self.remove_non_currency_sentences = remove_non_currency_sentences
        self.language = language

        self.train_mode = train_mode

    def run(self, corpus_tokenized: Union[Dict, Corpus]):

        if isinstance(corpus_tokenized, Dict):
            logger.info("Converting corpus_tokenized dict to Corpus object.")
            corpus = Corpus.from_dict(corpus_tokenized)
        else:
            corpus = corpus_tokenized

        logger.info("Tag currency tokens...")
        for document in tqdm(corpus):
            for segment in document:
                if isinstance(segment, Paragraph):
                    for sentence in segment:
                        tagger.tag_numeric_tokens(sentence=sentence, language=self.language)
                        tagger.tag_currency_tokens(sentence=sentence, language=self.language)
                elif isinstance(segment, Table):
                    tagger.tag_numeric_cells(table=segment, language=self.language)
                    tagger.tag_currency_cells(table=segment, language=self.language)

        if self.remove_non_currency_sentences:
            corpus = tagger.filter_corpus_by_ccy_sentences(corpus=corpus)

        if self.train_mode:
            self.save(corpus.to_dict(), "corpus_tagged", type_="pickle")
        else:
            return corpus
