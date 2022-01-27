from copy import deepcopy
import logging
from typing import List, Tuple, Dict, Optional, Union

from fluidml.common import Task
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

from kpi_relation_extractor.common.data_classes import Corpus
from kpi_relation_extractor.common.trainer.utils import set_seed_number, set_seeds


logger = logging.getLogger(__name__)


class SubWordTokenization(Task):

    publishes = ['corpus_tokenized', 'sub_word_tokenizer']

    def __init__(self,
                 special_tokens: Optional[Dict] = None,
                 seed: int = 42,
                 tokenizer_name: Optional[str] = None,
                 train_mode: bool = True):
        super().__init__()

        # config params
        self.special_tokens = special_tokens
        self.tokenizer_name = tokenizer_name
        self.seed = seed

        self.train_mode = train_mode

        self.sub_word_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        if self.special_tokens is not None:
            self.sub_word_tokenizer.add_special_tokens(self.special_tokens)

    def _get_pre_post_special_token_ids(self) -> Tuple[List[int], List[int]]:
        encoding_with_special_tokens = self.sub_word_tokenizer.encode('1', add_special_tokens=True)
        encoding_without_special_tokens = self.sub_word_tokenizer.encode('1', add_special_tokens=False)
        pre_special_token_ids, post_special_token_ids = [], []
        pre = True

        for x in encoding_with_special_tokens:
            if pre and x not in encoding_without_special_tokens:
                pre_special_token_ids.append(x)
            elif pre and x in encoding_without_special_tokens:
                pre = False
            elif not pre and x not in encoding_without_special_tokens:
                post_special_token_ids.append(x)
        return pre_special_token_ids, post_special_token_ids

    @staticmethod
    def _get_word2token_alignment_mask(word2token_start_indices: List[int],
                                       num_words: int,
                                       num_tokens: int) -> List[List[bool]]:
        word2token_alignment_mask = torch.zeros((num_words, num_tokens), dtype=torch.bool)
        for word_id, start in enumerate(word2token_start_indices):
            if word_id < len(word2token_start_indices) - 1:
                end = word2token_start_indices[word_id + 1]
                word2token_alignment_mask[word_id, start:end] = 1
            else:
                word2token_alignment_mask[word_id, start:] = 1

        return word2token_alignment_mask.tolist()

    def _tokenize_corpus(self,
                         corpus: Corpus) -> Corpus:
        pre_special_token_ids, post_special_token_ids = self._get_pre_post_special_token_ids()

        for sentence in tqdm(corpus.sentences):
            sentence.token_ids = deepcopy(pre_special_token_ids)

            word2token_start_ids = []
            word2token_end_ids = []
            start_index = len(pre_special_token_ids)
            words = [word.value for word in sentence.words]
            sentence_token_ids: List[List[int]] = self.sub_word_tokenizer.batch_encode_plus(
                words, add_special_tokens=False).input_ids
            for word_token_ids in sentence_token_ids:
                sentence.token_ids.extend(word_token_ids)

                word2token_start_ids.append(start_index)
                start_index += len(word_token_ids)
                word2token_end_ids.append(start_index - 1)

            sentence.word2token_alignment_mask = self._get_word2token_alignment_mask(
                word2token_start_ids,
                num_words=len(sentence.words),
                num_tokens=len(sentence.token_ids)
            )
            sentence.word2token_end_ids = word2token_end_ids
            sentence.word2token_start_ids = word2token_start_ids
            sentence.token_ids.extend(post_special_token_ids)
            sentence.tokens = self.sub_word_tokenizer.convert_ids_to_tokens(sentence.token_ids)

        return corpus

    def run(self, corpus_tagged: Union[Dict, Corpus]):
        set_seed_number(self.seed)
        set_seeds()

        if isinstance(corpus_tagged, Dict):
            logger.info('Converting corpus_tagged dict to Corpus object.')
            corpus = Corpus.from_dict(corpus_tagged)
        else:
            corpus = corpus_tagged

        logger.debug('Sub-word-tokenize corpus and calculate word to token alignment masks...')
        corpus_tokenized = self._tokenize_corpus(corpus)

        if self.train_mode:
            self.save(corpus_tokenized.to_dict(), 'corpus_tokenized', type_='pickle')
            self.save(self.sub_word_tokenizer, 'sub_word_tokenizer', type_='tokenizer')
        else:
            return corpus_tokenized, self.sub_word_tokenizer
