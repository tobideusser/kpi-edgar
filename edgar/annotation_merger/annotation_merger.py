import logging
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import xlrd
from tqdm import tqdm

from edgar import ALLOWED_RELATIONS, ALLOWED_ENTITIES, IS_ENTITY_NUMERIC
from edgar.data_classes import Corpus, Paragraph, Entity, Relation, Sentence, Labels


# https://stackoverflow.com/questions/64264563/attributeerror-elementtree-object-has-no-attribute-getiterator-when-trying
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True


logger = logging.getLogger(__name__)


class RelationError(Exception):
    """
    Exception raised if a relation occurs that should not be possible.
    """

    def __init__(self, entity1, entity2):
        self.message = f"{entity1['type']} is not allowed to be linked to {entity2['type']}!"


class AnnotationMerger:
    def __init__(
            self,
            excel_annotation_path: str,
            print_statistics: bool = False,
            ignore_noncritical_warnings: bool = False,
            skip_sentences_with_error: bool = False,
            filter_for_annotated_docs: bool = False,
            merge_auto_annotations: bool = False,
            label_mapping: Optional[Dict] = None
    ):
        self.excel_annotation_path = excel_annotation_path
        self.print_statistics = print_statistics
        self.ignore_noncritical_warnings = ignore_noncritical_warnings
        self.skip_sentences_with_error = skip_sentences_with_error
        self.filter_for_annotated_docs = filter_for_annotated_docs
        self.merge_auto_annotations = merge_auto_annotations
        self.label_mapping = label_mapping

    @staticmethod
    def _pairs_in_allowed_relations(entity_types: List) -> bool:
        """
        Generates all possible pairs from the list and checks if at least one
        of the pairs is in the ALLOWED_RELATIONS list.
        """
        pairs = [{x, y} for i1, x in enumerate(entity_types) for i2, y in enumerate(entity_types) if i1 < i2]
        for pair in pairs:
            if pair in ALLOWED_RELATIONS:
                return True
        return False

    @staticmethod
    def _isnumeric(string: str, characters_to_remove: str):
        for c in list(characters_to_remove):
            string = string.replace(c, "")
        return string.isnumeric()

    @staticmethod
    def _check_entity_link(entities: List, entity1_pos: int, entity2_pos: int) -> bool:
        """
        Checks if entity1 is linked to entity2. Raises an exception if the entities are not allowed
        to be linked together.
        """
        entity1 = entities[entity1_pos]
        entity2 = entities[entity2_pos]
        remaining_entities = [x["type"] for i, x in enumerate(entities)
                              if i not in {entity1_pos, entity2_pos}
                              and x["rel_anno1"] == entity1["rel_anno1"]
                              and x["rel_anno2"] == entity1["rel_anno2"]]
        if entity1["rel_anno1"] == entity2["rel_anno1"] and entity1["rel_anno1"] != "0":
            if entity1["rel_anno2"] == entity2["rel_anno2"] and entity1["rel_anno2"] is not None:
                if {entity1["type"], entity2["type"]} in ALLOWED_RELATIONS:
                    # example: davon_1_1 to davon_cy_1_1
                    return True
                elif (AnnotationMerger._pairs_in_allowed_relations(remaining_entities + [entity1["type"]])
                      or AnnotationMerger._pairs_in_allowed_relations(remaining_entities + [entity2["type"]])):
                    # example: davon_py_1_1 to davon_cy_1_1, but davon_1_1 exists
                    #          (and hence, davon_1_1 will be linked to davon_cy_1_1 and davon_py_1_1)
                    return False
                else:
                    # example: davon_1_1 to cy_1_1
                    #          davon is not allowed to be linked to cy -> annotation error
                    raise RelationError(entity1, entity2)
            else:
                if {entity1["type"], entity2["type"]} not in ALLOWED_RELATIONS:
                    if (
                            entity1["rel_anno2"] != entity2["rel_anno2"]
                            or AnnotationMerger._pairs_in_allowed_relations(remaining_entities + [entity1["type"]])
                            or AnnotationMerger._pairs_in_allowed_relations(remaining_entities + [entity2["type"]])
                    ):
                        # example1: kpi_1 to davon_cy_1_1
                        #           no annotation error, simply not linked
                        # example2: kpi_1 to davon_cy_1, but davon_1 exists
                        #           (and hence, davon_1 will be linked to davon_cy_1)
                        return False
                    else:
                        # example: kpi_1 to davon_cy_1
                        #          kpi is not allowed to be linked to davon_cy -> annotation error
                        raise RelationError(entity1, entity2)
                elif (
                        entity1["rel_anno2"] != entity2["rel_anno2"]
                        and entity1["rel_anno2"] is not None
                        and entity2["rel_anno2"] is not None
                ):
                    # example: davon_1_2 to davon_cy_1_1
                    #          no annotation error, simply not linked
                    return False
                else:
                    # example: kpi_1 to cy_1
                    #          or kpi_1 to davon_1_1
                    # todo: additional check: only certain entity types are allowed to
                    #       have the second relation annotation
                    return True
        else:
            # example: kpi_1 to kpi_2
            #          or kpi_1 to davon_2_1
            return False

    def _read_xl_annotations(self) -> List[Dict]:
        """
        Read annotations from Excel file and return a flattened list of dictionaries.
        """
        logger.info("Loading Excel workbook into memory...")
        wb = xlrd.open_workbook(self.excel_annotation_path)

        # todo: remove this "hack" and fully implement a recursive davon parsing
        reg_ex = re.compile(".*_[0-9]_[0-9]_[0-9]")

        logger.info("Extracting tokens, entities, and relations from each sheet...")
        results = []
        error_count = 0
        for sheet in tqdm(wb.sheets(), desc="Reading XL annotations"):
            if sheet.name not in ["legend", "possible_relations", "findings", "exercise", "solution"]:
                # doc name is in cell A1
                doc_name = sheet.cell(0, 0).value

                # split type should be in cell B1
                split_type = sheet.cell(0, 1).value
                split_type = None if split_type == "" else split_type

                try:
                    document_year = sheet.cell(0, 3).value
                except IndexError:
                    document_year = 2020

                # get amount of sentences stored in the sheet
                sentences_count = int((sheet.nrows - 1) / 7)

                for sentence_counter in range(sentences_count):
                    error_flag = False
                    skip_flag = False

                    # read blob id and sentence id of current sentence
                    blob_id = int(sheet.cell(sentence_counter * 7 + 2, 1).value)
                    sentence_id = int(sheet.cell(sentence_counter * 7 + 3, 1).value)

                    # read the actual sentence tokens & raw annotations
                    # todo: str(token)
                    tokens = [str(token) for token in sheet.row_values(sentence_counter * 7 + 4)[1:] if token != ""]
                    raw_auto_annotations = [anno for anno in
                                            sheet.row_values(sentence_counter * 7 + 5)[1:(len(tokens) + 1)]]
                    raw_man_annotations = [anno for anno in
                                           sheet.row_values(sentence_counter * 7 + 6)[1:(len(tokens) + 1)]]

                    # merge auto annotations and manual annotations
                    if self.merge_auto_annotations:
                        raw_annotations = []
                        for auto_anno, man_anno in zip(raw_auto_annotations, raw_man_annotations):
                            if man_anno == "":
                                raw_annotations.append(auto_anno)
                            elif man_anno == "x":
                                raw_annotations.append("")
                            else:
                                raw_annotations.append(man_anno)
                    else:
                        raw_annotations = raw_man_annotations

                    # todo: ugly skip of nested davons, fully parse them
                    if len(list(filter(reg_ex.match, raw_annotations))) == 0:
                        # get list of unique entities
                        # todo: sorted
                        raw_entities = sorted(set(raw_annotations))

                        # save start and end of entities
                        entities = []
                        for raw_entity in raw_entities:
                            if raw_entity != "":
                                # get all positions of the entity
                                positions = [pos for pos, raw_anno in enumerate(raw_annotations)
                                             if raw_anno == raw_entity]

                                if raw_entity == "false_positive":
                                    entities.append({"type": raw_entity,
                                                     "start": min(positions),
                                                     "end": max(positions) + 1, "rel_anno1": "0",
                                                     "rel_anno2": None})
                                else:
                                    # remove relation identifier (e.g. 'kpi_1' becomes 'kpi')
                                    # and map to original entity_type
                                    entity_splitted = raw_entity.split("_")
                                    rel_annos = [rel_anno for rel_anno in entity_splitted if rel_anno.isnumeric()]
                                    entity_type = "_".join([et for et in entity_splitted if not et.isnumeric()])

                                    if sorted(positions) != list(range(min(positions), max(positions) + 1)):

                                        logger.warning("Non Consecutive Annotation Warning")
                                        logger.warning(f"The annotation {raw_entity} appears in non consecutive cells.")
                                        logger.warning(f" Location: sheet_name={sheet.name}, doc_name={doc_name}, "
                                                       f"blob_id={blob_id}, sentence_id={sentence_id}")
                                        error_count += 1
                                        error_flag = True

                                    elif entity_type not in ALLOWED_ENTITIES:

                                        if entity_type == "delete":
                                            skip_flag = True
                                        else:

                                            logger.warning("Entity Type Warning")
                                            logger.warning(f"Found entity type {entity_type} which is currently not "
                                                           f"supported.")
                                            logger.warning(f" Location: sheet_name={sheet.name}, doc_name={doc_name}, "
                                                           f"blob_id={blob_id}, sentence_id={sentence_id}")
                                            error_count += 1
                                            error_flag = True

                                    elif (AnnotationMerger._isnumeric(
                                            " ".join(map(str, tokens[min(positions):(max(positions) + 1)])), "-., "
                                            ) != IS_ENTITY_NUMERIC[entity_type]
                                          and not self.ignore_noncritical_warnings):

                                        logger.warning("Entity Value Warning (non-critical)")
                                        if IS_ENTITY_NUMERIC[entity_type]:
                                            logger.warning(f"Found entity type {entity_type} with value "
                                                           f"'{' '.join(tokens[min(positions):(max(positions) + 1)])}' "
                                                           "which is not a numeric value but should be.")
                                        else:
                                            logger.warning(f"Found entity type {entity_type} with value "
                                                           f"'{' '.join(tokens[min(positions):(max(positions) + 1)])}' "
                                                           "which is a numeric value but should not be.")
                                        logger.warning(f" Location: sheet_name={sheet.name}, doc_name={doc_name}, "
                                                       f"blob_id={blob_id}, sentence_id={sentence_id}")
                                        error_count += 1
                                        error_flag = True

                                    else:

                                        try:
                                            entities.append(
                                                {"type": entity_type,
                                                 "start": min(positions),
                                                 "end": max(positions) + 1,
                                                 "rel_anno1": rel_annos[0],
                                                 "rel_anno2": None if len(rel_annos) == 1 else rel_annos[1]}
                                            )
                                        except IndexError:
                                            logger.warning("No Relation Annotation Warning")
                                            logger.warning(f"The entity {raw_entity} likely has no relation annotation.")
                                            logger.warning(f" Location: sheet_name={sheet.name}, doc_name={doc_name}, "
                                                           f"blob_id={blob_id}, sentence_id={sentence_id}")
                                            error_count += 1
                                            error_flag = True

                        # save head & tail of relations
                        relations = []
                        for i, entity1 in enumerate(entities):
                            for j, entity2 in enumerate(entities):
                                try:
                                    # reason for i < j:
                                    #   stops relation from appearing twice (and stops linking to itself)
                                    if i < j and AnnotationMerger._check_entity_link(entities, i, j):
                                        # todo: if statement correct?
                                        # relations.append({"type": "matches", "head": i, "tail": j})
                                        if entity1['start'] < entity2['start']:
                                            relations.append({"type": "matches", "head": i, "tail": j})
                                        else:
                                            relations.append({"type": "matches", "head": j, "tail": i})
                                except RelationError as err:
                                    logger.warning("Relation Not Allowed Warning")
                                    logger.warning(err.message)
                                    logger.warning(f" Location: sheet_name={sheet.name}, doc_name={doc_name}, "
                                                   f"blob_id={blob_id}, sentence_id={sentence_id}")
                                    # logger.warning("Please fix this warning, as it will force mistakes later on.\n")
                                    error_count += 1
                                    error_flag = True

                        # remove rel_anno from entity dict
                        for entity in entities:
                            del entity["rel_anno1"]
                            del entity["rel_anno2"]

                        if error_flag and self.skip_sentences_with_error:
                            logger.warning("Error found, sentence was skipped.\n")
                        elif not skip_flag:
                            # add the extracted information as a dict to the result list
                            results.append(
                                {
                                    "tokens": tokens,
                                    "entities": entities,
                                    "relations": relations,
                                    "doc_name": doc_name,
                                    "segment_id": blob_id,
                                    "sentence_id": sentence_id,
                                    "split_type": split_type,
                                    "document_year": document_year
                                }
                            )
                    else:
                        logger.warning("Nested relation found, skipping sentence.")
                        logger.warning(f" Location: sheet_name={sheet.name}, doc_name={doc_name}, "
                                       f"blob_id={blob_id}, sentence_id={sentence_id}")
        if self.print_statistics:
            logger.warning("\nGeneral Information:\n")
            logger.warning(f"Documents annotated: {len(np.unique([res['doc_name'] for res in results]))}")
            logger.warning(f"Sentences annotated: {len(results)}")
            sentences_split = {split_type: [
                sentence for sentence in results if sentence["split_type"] == split_type
            ] for split_type in ["train", "valid", "test"]}
            for split_type in ["train", "valid", "test"]:
                logger.warning(f" - thereof {split_type}: {len(sentences_split[split_type])}")
            # extract all entities
            all_entities = [entity["type"]
                            for sentence in [sentence["entities"] for sentence in results]
                            for entity in sentence]
            logger.warning(f"Entities annotated: {len(all_entities)}")
            # extract all relations
            all_relations = []
            for res in results:
                if len(res["relations"]) > 0:
                    for relation in res["relations"]:
                        entity1_type = res["entities"][relation["head"]]["type"]
                        entity2_type = res["entities"][relation["tail"]]["type"]
                        all_relations.append(" - ".join(sorted([entity1_type, entity2_type])))
            logger.warning(f"Relations annotated: {len(all_relations)}")
            logger.warning(f"Annotation warnings: {error_count}")

            # Get unique entities and count of each
            (unique_entities, counts) = np.unique(all_entities, return_counts=True)
            count_sort_ind = np.argsort(-counts)
            logger.warning("\nEntity Frequency:\n")
            for unique_entity, count in zip(unique_entities[count_sort_ind].tolist(), counts[count_sort_ind].tolist()):
                logger.warning(f"{unique_entity}: {count} ({round((count / sum(counts)) * 100, 2)}%)")

            # Get unique relations and count of each
            (unique_relations, counts) = np.unique(all_relations, return_counts=True)
            count_sort_ind = np.argsort(-counts)
            logger.warning("\nRelation Frequency:\n")
            for unique_relation, count in zip(unique_relations[count_sort_ind].tolist(),
                                              counts[count_sort_ind].tolist()):
                logger.warning(f"{unique_relation}: {count} ({round((count / sum(counts)) * 100, 2)}%)")
            logger.warning("\n")
        return results

    def _merge_xl_to_corpus(self, corpus: Corpus) -> Corpus:

        annotations = self._read_xl_annotations()

        # list of all document names in corpus for easier access later on
        all_doc_names_corpus = [doc.id_ for doc in corpus.documents]

        # put all sentences from the input that have the same doc in a dict
        all_annotated_docs = {}
        for sentence in annotations:
            # create entry if it does not exist
            if sentence["doc_name"] not in all_annotated_docs:
                all_annotated_docs[sentence["doc_name"]] = []
            # add sentence to the list
            all_annotated_docs[sentence["doc_name"]].append(sentence)

        # add annotations to each sentence in each document (if available)
        logger.info("Adding annotated sentences to corpus...")
        for annotated_doc_name, annotated_doc in tqdm(all_annotated_docs.items(), desc="Adding annotations"):
            # check if name of annotated document appears in Corpus
            if annotated_doc_name not in all_doc_names_corpus:
                logger.warning(f"\nThe annotated document {annotated_doc_name} was not found in the corpus.")
                logger.warning(f"Was the wrong or incomplete corpus loaded?")
            else:
                corpus_doc_id = all_doc_names_corpus.index(annotated_doc_name)
                corpus.documents[corpus_doc_id].is_annotated = True
                corpus_segment_ids = [segment.id_ for segment in corpus.documents[corpus_doc_id].segments]
                for annotated_sentence in annotated_doc:

                    if len(annotated_sentence["entities"]) > 0:

                        # pos_in_segment_list & pos_in_sentence_list is required because the
                        # corresponding IDs are not equal to the position of the blob or sentence in its list
                        pos_in_segment_list = corpus_segment_ids.index(annotated_sentence["segment_id"])
                        segment_sentence_ids = [segment.id_ for segment in
                                                corpus.documents[corpus_doc_id].segments[pos_in_segment_list].sentences]
                        pos_in_sentence_list = segment_sentence_ids.index(annotated_sentence["sentence_id"])
                        # loop through all entities in the annotated sentence and create Entity class objects for each
                        entities = [Entity(type_=entity["type"], start=entity["start"], end=entity["end"])
                                    for entity in annotated_sentence["entities"]]
                        # add all Entity objects to the sentence in the corpus
                        corpus.documents[corpus_doc_id].segments[pos_in_segment_list].sentences[
                            pos_in_sentence_list].entities_anno = entities

                        # add split type
                        corpus.documents[corpus_doc_id].segments[pos_in_segment_list].sentences[
                            pos_in_sentence_list].split_type = annotated_sentence["split_type"]

                        if len(annotated_sentence["relations"]) > 0:

                            # loop through all relations in the annotated sentence
                            # and create Relation class objects for each
                            relations = [Relation(type_=relation["type"],
                                                  head_idx=relation["head"],
                                                  tail_idx=relation["tail"])
                                         for relation in annotated_sentence["relations"]]
                            # add all Relation objects to the sentence in the corpus
                            corpus.documents[corpus_doc_id].segments[pos_in_segment_list].sentences[
                                pos_in_sentence_list].relations_anno = relations

        if self.filter_for_annotated_docs:
            corpus = AnnotationMerger.filter_annotated_samples(corpus)
        return corpus

    @staticmethod
    def filter_annotated_samples(corpus: Corpus) -> Corpus:
        docs_to_keep = []
        for i, document in enumerate(corpus.documents):
            if document.is_annotated:
                docs_to_keep.append(i)
                for segment in document.segments:
                    if isinstance(segment, Paragraph):
                        segment.sentences = [sentence for sentence in segment.sentences if sentence.entities_anno]

        corpus.documents = [document for i, document in enumerate(corpus.documents) if i in docs_to_keep]
        return corpus

    @staticmethod
    def tag_iobes(sentence: Sentence, labels: Labels):
        iobes = ["O"] * len(sentence)

        # if sentence.entities_anno:
        for ent in sentence.entities_anno:
            if ent.end - ent.start == 1:
                iobes[ent.start] = f"S-{ent.type_}"

            else:
                iobes[ent.start] = f"B-{ent.type_}"
                iobes[ent.end - 1] = f"E-{ent.type_}"

                iobes[ent.start + 1: ent.end - 1] = [f"I-{ent.type_}"] * (ent.end - ent.start - 2)

        sentence.entities_anno_iobes = iobes
        sentence.entities_anno_iobes_ids = [labels.iobes.val2idx[label] for label in iobes]

    @staticmethod
    def _add_iobes_annotations(corpus: Corpus, labels: Labels) -> Corpus:
        for sentence in tqdm(corpus.sentences, desc="Adding IOBES tags"):
            if sentence.entities_anno:
                AnnotationMerger.tag_iobes(sentence, labels)
        return corpus

    def _map_annotations(self, corpus: Corpus) -> Corpus:
        for sentence in corpus.sentences:
            if sentence.entities_anno:
                # apply label mapping. if mapped label is None, delete corresponding entity
                entity_ids_to_keep = []
                for idx, entity in enumerate(sentence.entities_anno):
                    entity.type_ = self.label_mapping[entity.type_]
                    if entity.type_:
                        entity_ids_to_keep.append(idx)

                sentence.entities_anno = [ent for idx, ent in enumerate(sentence.entities_anno)
                                          if idx in entity_ids_to_keep]
            if sentence.relations_anno:
                # correct head_idx and tail_idx due to deleted entities
                # delete relations which hold a deleted entity
                relation_ids_to_keep = []
                for idx, relation in enumerate(sentence.relations_anno):
                    if relation.head_idx in entity_ids_to_keep and relation.tail_idx in entity_ids_to_keep:
                        relation.head_idx = entity_ids_to_keep.index(relation.head_idx)
                        relation.tail_idx = entity_ids_to_keep.index(relation.tail_idx)
                        relation_ids_to_keep.append(idx)

                sentence.relations_anno = [rel for idx, rel in enumerate(sentence.relations_anno)
                                           if idx in relation_ids_to_keep]
            if not sentence.entities_anno and not sentence.relations_anno:
                sentence.split_type = None
        return corpus

    def merge_annotations(self, corpus: Corpus) -> Tuple[Corpus, Labels]:
        corpus = self._merge_xl_to_corpus(corpus)
        if self.label_mapping is not None:
            corpus = self._map_annotations(corpus)
        labels = Labels.from_corpus(corpus)
        corpus = AnnotationMerger._add_iobes_annotations(corpus, labels)
        return corpus, labels

