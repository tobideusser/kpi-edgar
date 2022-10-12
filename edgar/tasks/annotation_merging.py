import logging
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
    ):
        super().__init__()

        self.excel_annotation_file = excel_annotation_file

        self.train_mode = train_mode
        self.ignore_noncritical_warnings = ignore_noncritical_warnings
        self.skip_sentences_with_error = skip_sentences_with_error
        self.filter_for_annotated_docs = filter_for_annotated_docs
        self.label_mapping = label_mapping
        self.parse_secondary_annotations = parse_secondary_annotations

    def run(self, corpus_tagged: Union[Dict, Corpus]):

        if isinstance(corpus_tagged, Dict):
            logger.info("Converting corpus_tagged dict to Corpus object.")
            corpus = Corpus.from_dict(corpus_tagged)
        else:
            corpus = corpus_tagged

        if self.train_mode:

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

            self.save(corpus.to_dict(), "corpus_tagged", type_="pickle")
            self.save(labels.to_dict(), "labels", type_="json")
        else:
            return corpus
