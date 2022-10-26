import logging
from copy import deepcopy
from typing import Dict, Union, Optional

from fluidml import Task

from edgar.annotation_merger import AnnotationMerger
from edgar.data_classes import Corpus

logger = logging.getLogger(__name__)


class AnnotationMerging(Task):

    publishes = ["corpus_tagged", "labels"]

    def __init__(
        self,
        excel_annotation_file: str,
        train_mode: bool = True,
        ignore_noncritical_warnings: bool = False,
        skip_sentences_with_error: bool = False,
        filter_for_annotated_docs: bool = False,
        label_mapping: Optional[Dict] = None,
        parse_secondary_annotations: bool = False,
        save_as_json: bool = False,
    ):
        super().__init__()

        self.excel_annotation_file = excel_annotation_file

        self.train_mode = train_mode
        self.ignore_noncritical_warnings = ignore_noncritical_warnings
        self.skip_sentences_with_error = skip_sentences_with_error
        self.filter_for_annotated_docs = filter_for_annotated_docs
        self.label_mapping = label_mapping
        self.parse_secondary_annotations = parse_secondary_annotations
        self.save_as_json = save_as_json

    def _save_as_json(self, corpus: Corpus):
        corpus_dict = deepcopy(corpus).to_dict()["documents"]

        for doc in corpus_dict:
            # clean-up dict, i.e. delete not yet used entries
            del doc["document_year"]  # todo: parse the document year
            del doc["is_annotated"]
            for segment in doc["segments"]:
                if segment["sentences"] is None:
                    del segment
                else:
                    del segment["textblock_entity"]  # edgar specific entities
                    del segment["edgar_entities"]  # edgar specific entities
                    del segment["tag"]
                    for sentence in segment["sentences"]:
                        del sentence["edgar_entities"]  # edgar specific entities
                        del sentence["tokens"]
                        del sentence["token_ids"]
                        del sentence["word2token_alignment_mask"]
                        del sentence["word2token_start_ids"]
                        del sentence["word2token_end_ids"]
                        del sentence["entities_pred"]
                        del sentence["relations_pred"]
                        del sentence["entities_anno_gold"]
                        del sentence["relations_anno_gold"]

        self.save(corpus_dict, "kpi_edgar", type_="json")

    def run(self, corpus_tagged: Union[Dict, Corpus]):

        if isinstance(corpus_tagged, Dict):
            logger.info("Converting corpus_tagged dict to Corpus object.")
            corpus = Corpus.from_dict(corpus_tagged)
        else:
            corpus = corpus_tagged

        annotation_merger = AnnotationMerger(
            excel_annotation_path=self.excel_annotation_file,
            ignore_noncritical_warnings=self.ignore_noncritical_warnings,
            skip_sentences_with_error=self.skip_sentences_with_error,
            filter_for_annotated_docs=self.filter_for_annotated_docs,
            label_mapping=self.label_mapping,
            print_statistics=True,
            parse_secondary_annotations=self.parse_secondary_annotations,
        )

        corpus, labels = annotation_merger.merge_annotations(corpus)

        if self.save_as_json:
            self._save_as_json(corpus=corpus)

        if self.train_mode:
            self.save(corpus.to_dict(), "corpus_tagged", type_="pickle")
            self.save(labels.to_dict(), "labels", type_="json")
        else:
            return corpus
