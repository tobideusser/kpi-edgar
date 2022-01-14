import logging
import os
from typing import List, Tuple, Iterator

from syntok.tokenizer import Tokenizer as SynTokenizer
from tqdm import tqdm

from kpi_relation_extractor import package_path
from kpi_relation_extractor.common.data_classes import Corpus, Paragraph, Table, Sentence, Word
from kpi_relation_extractor.common.tokenizer.sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(
            self,
            remove_tables: bool = False,
            language: str = "en"
    ):
        self._remove_tables = remove_tables

        self._tokenizer = SynTokenizer()
        self._sent_splitter = SentenceSplitter(
            language=language,
            non_breaking_prefix_file=os.path.join(
                package_path, "common", "tokenizer", f"{language}_sentencesplitter_additional.txt"
            )
        )

    @staticmethod
    def _join_hyphen_tokens(gen_tokens: Iterator) -> List[Tuple[str, int]]:
        tokens = []
        flag = False
        for token in gen_tokens:
            if flag:
                del tokens[-1]
                tokens.append(('-' + token.value, token.offset - 1))
                flag = False
            else:
                if token.spacing and token.spacing != ' ' and tokens:
                    prev_token = tokens[-1][0]
                    del tokens[-1]
                    tokens.append(
                        (prev_token + token.spacing + token.value, token.offset - len(prev_token + token.spacing)))
                else:
                    tokens.append((token.value, token.offset))

            if token.value == '-' and token.spacing == ' ':
                flag = True
        return tokens

    def tokenize(self, text: str) -> List[Word]:
        gen_tokens = self._tokenizer.tokenize(text)
        tokens = Tokenizer._join_hyphen_tokens(gen_tokens=gen_tokens)

        tokens = [Word(id_=id_, value=token[0]) for id_, token in enumerate(tokens)]
        return tokens

    def split_text_in_sentences(self, text: str, document_id: str, segment_id: int) -> List[Sentence]:
        sentences = self._sent_splitter.split(text)
        sentences = [Sentence(id_=id_, value=sentence) for id_, sentence in enumerate(sentences)]
        for sentence in sentences:
            # split sentence in words
            sentence.words = self.tokenize(sentence.value)
            # set unique sentence id (unique within corpus)
            sentence.unique_id = f'{document_id}_{segment_id}_{sentence.id_}'
        return sentences

    def tokenize_corpus(self,
                        corpus: Corpus):
        if self._remove_tables:
            for document in tqdm(corpus):
                segments = []
                for segment in document:
                    if isinstance(segment, Paragraph):
                        segment.sentences = self.split_text_in_sentences(segment.value, document.id_, segment.id_)
                        segments.append(segment)

                document.segments = segments

        else:
            for document in tqdm(corpus):
                for segment in document:

                    if isinstance(segment, Paragraph):
                        segment.sentences = self.split_text_in_sentences(segment.value, document.id_, segment.id_)

                    elif isinstance(segment, Table):
                        for cell in segment:
                            cell.words = self.tokenize(cell.value)
        return corpus
