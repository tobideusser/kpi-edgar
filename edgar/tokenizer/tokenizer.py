import logging
import os
from typing import List, Tuple, Iterator

from syntok.tokenizer import Tokenizer as SynTokenizer
from tqdm import tqdm

from edgar import package_path
from edgar.data_classes import Corpus, Paragraph, Table, Sentence, Word, Segment, EdgarEntity
from edgar.tokenizer import SentenceSplitter

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, remove_tables: bool = False, language: str = "en"):
        self._remove_tables = remove_tables

        self._tokenizer = SynTokenizer()
        self._sent_splitter = SentenceSplitter(
            language=language,
            non_breaking_prefix_file=os.path.join(
                package_path, "tokenizer", f"{language}_sentencesplitter_additional.txt"
            ),
        )

    @staticmethod
    def _join_hyphen_tokens(gen_tokens: Iterator) -> List[Tuple[str, int]]:
        tokens = []
        flag = False
        for token in gen_tokens:
            if flag:
                del tokens[-1]
                tokens.append(("-" + token.value, token.offset - 1))
                flag = False
            else:
                if token.spacing and token.spacing != " " and tokens:
                    prev_token = tokens[-1][0]
                    del tokens[-1]
                    tokens.append(
                        (prev_token + token.spacing + token.value, token.offset - len(prev_token + token.spacing))
                    )
                else:
                    tokens.append((token.value, token.offset))

            if token.value == "-" and token.spacing == " ":
                flag = True
        return tokens

    def tokenize(self, sentence: Sentence, text_len: int, edgar_entities: List[EdgarEntity]) -> Tuple[Sentence, int]:
        gen_tokens = self._tokenizer.tokenize(sentence.value)
        tokens = Tokenizer._join_hyphen_tokens(gen_tokens=gen_tokens)

        # initialise the list of words in the Sentence object that will be filled
        sentence.words = []

        if edgar_entities:

            # entity_start and entity_end hold the start and end character positions of entities
            entity_start = [entity.start_char for entity in edgar_entities]
            entity_end = [entity.end_char for entity in edgar_entities]

            # a flip switch that will signify if an entity have been found and is not yet completed
            # if True, will look for the start of an entity
            # if False, will look for the end of an entity
            start_end_flag = True

            # will hold the position of the found entity in the list edgar_entities
            entity_sentence_id = -1

            for id_, token in enumerate(tokens):
                if start_end_flag and text_len + token[1] in entity_start:
                    entity_sentence_id = entity_start.index(text_len + token[1])
                    # set the start_word integer to the position of the first word of the entity in the sentence
                    edgar_entities[entity_sentence_id].start_word = id_
                    start_end_flag = False
                elif not start_end_flag and text_len + token[1] >= entity_end[entity_sentence_id]:
                    # set the end_word integer to the position of the current word / the end of the entity
                    edgar_entities[entity_sentence_id].end_word = id_
                    start_end_flag = True
                    if not sentence.edgar_entities:
                        # if sentence.edgar_entities is not yet initialised, do it now
                        sentence.edgar_entities = []
                    # append the EdgarEntity (now with a filled in start and end word id) to the Sentence object
                    sentence.edgar_entities.append(edgar_entities[entity_sentence_id])
                sentence.words.append(Word(id_=id_, value=token[0]))

        else:
            sentence.words = [Word(id_=id_, value=token[0]) for id_, token in enumerate(tokens)]
        # new text_len will be:
        #   previous text_len
        #   + the length of all previous characters upto the last one -> tokens[-1][1]
        #   + the length of the last token -> len(tokens[-1][0])
        #   + 1 for the trailing space
        text_len += tokens[-1][1] + len(tokens[-1][0]) + 1
        return sentence, text_len

    def split_text_in_sentences(self, segment: Segment, document_id: str, segment_id: int) -> List[Sentence]:
        sentences = self._sent_splitter.split(segment.value)
        sentences = [Sentence(id_=id_, value=sentence) for id_, sentence in enumerate(sentences)]
        text_len = 0
        for sentence in sentences:
            # split sentence in words
            sentence, text_len = self.tokenize(
                sentence=sentence, text_len=text_len, edgar_entities=segment.edgar_entities
            )
            # set unique sentence id (unique within corpus)
            sentence.unique_id = f"{document_id}_{segment_id}_{sentence.id_}"
        return sentences

    def tokenize_corpus(self, corpus: Corpus):
        if self._remove_tables:
            for document in tqdm(corpus):
                segments = []
                for segment in document:
                    if isinstance(segment, Paragraph):
                        segment.sentences = self.split_text_in_sentences(segment, document.id_, segment.id_)
                        segments.append(segment)

                document.segments = segments

        else:
            for document in tqdm(corpus):
                for segment in document:

                    if isinstance(segment, Paragraph):
                        segment.sentences = self.split_text_in_sentences(segment, document.id_, segment.id_)

                    elif isinstance(segment, Table):
                        for cell in segment:
                            cell.words = self.tokenize(cell)
        return corpus
