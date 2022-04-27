from copy import deepcopy
from typing import List, Optional, Dict, Set

import numpy as np
import torch

from edgar.trainer.metrics import Metric
from edgar.trainer.utils import nan_safe_tensor_divide


class FBeta(Metric):
    """Compute precision, recall, F-measure and support for each class.
    The precision is the ratio `tp / (tp + fp)` where `tp` is the number of
    true positives and `fp` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio `tp / (tp + fn)` where `tp` is the number of
    true positives and `fn` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.
    If we have precision and recall, the F-beta score is simply:
    `F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)`
    The F-beta score weights recall more than precision by a factor of
    `beta`. `beta == 1.0` means recall and precision are equally important.
    The support is the number of occurrences of each class in `y_true`.
    # Parameters
    beta : `float`, optional (default = `1.0`)
        The strength of recall versus precision in the F-score.
    average : `str`, optional (default = `None`)
        If `None`, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        `'micro'`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        `'macro'`:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.
        `'weighted'`:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
    labels: `list`, optional
        The set of labels to include and their order if `average is None`.
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro or weighted average.
    """

    def __init__(self, beta: float = 1.0, average: str = None, labels: List[int] = None) -> None:
        super().__init__()
        average_options = {None, "micro", "macro", "weighted"}
        if average not in average_options:
            raise ValueError(f"`average` has to be one of {average_options}.")
        if beta <= 0:
            raise ValueError("`beta` should be >0 in the F-beta score.")
        if labels is not None and len(labels) == 0:
            raise ValueError("`labels` cannot be an empty list.")

        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Optional[torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Optional[torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Optional[torch.Tensor] = None

    def __call__(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        targets : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, targets = self.detach_tensors(predictions, targets)
        device = targets.device

        # Calculate true_positive_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        if (targets >= num_classes).any():
            raise ValueError(
                "A gold label passed to FBetaMeasure contains "
                f"an id >= {num_classes}, the number of classes."
            )

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(targets).bool()
        targets = targets.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0
        argmax_predictions = predictions.max(dim=-1)[1].float()

        true_positives = (targets == argmax_predictions) & mask & pred_mask
        true_positives_bins = targets[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=device)
        else:
            true_positive_sum = torch.bincount(
                true_positives_bins.long(), minlength=num_classes
            ).float()

        pred_bins = argmax_predictions[mask & pred_mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=device)

        targets_bins = targets[mask].long()
        if targets.shape[0] != 0:
            true_sum = torch.bincount(targets_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=predictions.device)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    def get_metric(self, reset: bool = False):
        """
        # Returns
        precisions : `List[float]`
        recalls : `List[float]`
        f-measures : `List[float]`
        !!! Note
            If `self.average` is not `None`, you will get `float` instead of `List[float]`.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        else:
            tp_sum = self._true_positive_sum
            pred_sum = self._pred_sum
            true_sum = self._true_sum

        if self._labels is not None:
            # Retain only selected labels and order them
            tp_sum = tp_sum[self._labels]
            pred_sum = pred_sum[self._labels]  # type: ignore
            true_sum = true_sum[self._labels]  # type: ignore

        if self._average == "micro":
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()  # type: ignore
            true_sum = true_sum.sum()  # type: ignore

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = nan_safe_tensor_divide(tp_sum, pred_sum)
        recall = nan_safe_tensor_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0

        if self._average == "macro":
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        elif self._average == "weighted":
            weights = true_sum
            weights_sum = true_sum.sum()  # type: ignore
            precision = nan_safe_tensor_divide((weights * precision).sum(), weights_sum)
            recall = nan_safe_tensor_divide((weights * recall).sum(), weights_sum)
            fscore = nan_safe_tensor_divide((weights * fscore).sum(), weights_sum)

        if reset:
            self.reset()

        if self._average is None:
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "fscore": fscore.tolist(),
            }
        else:
            return {"precision": precision.item(), "recall": recall.item(), "fscore": fscore.item()}

    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None


class F1(FBeta):
    """
    Computes Precision, Recall and F1. Same as FBeta but fixes beta=1.
    """

    def __init__(self, average: str = None, labels: List[int] = None) -> None:
        super().__init__(beta=1, average=average, labels=labels)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns
        precision : `float`
        recall : `float`
        f1-measure : `float`
        """
        metric = super().get_metric(reset=reset)
        precision = metric["precision"]
        recall = metric["recall"]
        f1 = metric["fscore"]
        return {"precision": precision, "recall": recall, "f1": f1}


class NERF1Adjusted(Metric):
    def __init__(self):
        super().__init__()

        self.pred_entities: List[List[Dict]] = []
        self.gt_entities: List[List[Dict]] = []
        self.entity_types: Optional[List[str]] = None

    def __call__(self,
                 entities_anno: List[List[Dict]],
                 entities_pred: List[Dict],
                 entity_types: List[str]):
        """Evaluate NER predictions
                Args:
                    pred_entities (list) :  list of list of predicted entities (several entities in each sentence)
                    gt_entities (list) :    list of list of ground truth entities
                        entity = {"start": start_idx (inclusive),
                                  "end": end_idx (exclusive),
                                  "type": ent_type}
                    entity_types (list):     list of entity types
                                  """
        if self.entity_types is None:
            self.entity_types = entity_types

        self.gt_entities.extend(entities_anno)
        # self.pred_entities.extend(pred_entities)
        self.pred_entities.extend([[{"start": span[0], "end": span[1], "type_": ent_type}
                                    for span, ent_type in s.items()]
                                   for s in entities_pred])

    def get_metric(self, reset: bool = False):
        assert len(self.pred_entities) == len(self.gt_entities)

        statistics = {
            ent: {
                "support": 0,
                # Strict: exact boundary surface string match and entity type
                "strict": {"tp": 0, "fp": 0, "fn": 0},
                # Partial & Type: some overlap between the system tagged entity and the gold annotation
                "partial_type": {"tp": 0, "fp": 0, "fn": 0},
                # Exact: exact boundary match over the surface string, regardless of the type
                "exact": {"tp": 0, "fp": 0, "fn": 0},
                # Partial: partial boundary match over the surface string, regardless of the type
                "partial": {"tp": 0, "fp": 0, "fn": 0}
            } for ent in self.entity_types
        }
        clf_report = {}

        # Count GT entities and Predicted entities
        # n_sents = len(self.gt_entities)
        # n_phrases = sum([len([ent for ent in sent]) for sent in self.gt_entities])
        # n_found = sum([len([ent for ent in sent]) for sent in self.pred_entities])

        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(self.pred_entities, self.gt_entities):
            # pred_ents = {(ent["start"], ent["end"]) for ent in pred_sent if ent["type_"] == ent_type}
            # gt_ents = {(ent["start"], ent["end"]) for ent in gt_sent if ent["type_"] == ent_type}

            # TP and FN by looking from ground truth
            for ground_truth_entity in gt_sent:
                gt_ent_type = ground_truth_entity["type_"]
                # gt_ent_span = [ground_truth_entity["start"], ground_truth_entity["end"]]
                ground_truth_entity_span = set(range(ground_truth_entity["start"], ground_truth_entity["end"]))
                length_ground_truth_entity = ground_truth_entity["end"] - ground_truth_entity["start"]

                # measures how much of the entity was already predicted with the correct type
                largest_partial_type_tp_overlap = 0
                corresponding_partial_type_fp_overlap = 0
                # measures how much of the entity was already predicted regardless of type
                largest_partial_tp_overlap = 0
                corresponding_partial_fp_overlap = 0

                statistics[gt_ent_type]["support"] += 1

                best_strict_entity = None
                best_partial_type_entity = None
                best_exact_entity = None
                best_partial_entity = None

                for predicted_entity in pred_sent:
                    predicted_entity_span = set(range(predicted_entity["start"], predicted_entity["end"]))
                    raw_overlap = predicted_entity_span.intersection(ground_truth_entity_span)
                    fp_overlap = len(predicted_entity_span - raw_overlap) / len(predicted_entity_span)
                    tp_overlap = len(raw_overlap) / length_ground_truth_entity

                    # 5 possible cases (as seen in variable statistics definition):
                    #   case 1: exact boundary surface string match and entity type
                    #   case 2: some overlap between the system tagged entity and the gold annotation
                    #   case 3: exact boundary match over the surface string, regardless of the type
                    #   case 4: partial boundary match over the surface string, regardless of the type
                    #   case 5: no match at all
                    if predicted_entity["type_"] == gt_ent_type:
                        if tp_overlap == 1 and fp_overlap == 0:
                            # case 1
                            statistics[gt_ent_type]["strict"]["tp"] += 1
                            statistics[gt_ent_type]["exact"]["tp"] += 1
                            largest_partial_type_tp_overlap = 1
                            corresponding_partial_type_fp_overlap = 0
                            largest_partial_tp_overlap = 1
                            corresponding_partial_fp_overlap = 0
                            best_strict_entity = predicted_entity
                            best_partial_type_entity = predicted_entity
                            best_exact_entity = predicted_entity
                            best_partial_entity = predicted_entity
                            break  # can break loop, since all other predicted entities will be inferior
                        else:
                            if tp_overlap > largest_partial_type_tp_overlap:
                                # case 2
                                largest_partial_type_tp_overlap = tp_overlap
                                corresponding_partial_type_fp_overlap = fp_overlap
                                best_partial_type_entity = predicted_entity
                            if tp_overlap > largest_partial_tp_overlap:
                                # case 4
                                largest_partial_tp_overlap = tp_overlap
                                corresponding_partial_fp_overlap = fp_overlap
                                best_partial_entity = predicted_entity
                    else:
                        if tp_overlap == 1 and fp_overlap == 0:
                            # case 3
                            largest_partial_tp_overlap = 1
                            corresponding_partial_fp_overlap = 0
                            best_exact_entity = predicted_entity
                            best_partial_entity = predicted_entity
                        elif tp_overlap > largest_partial_tp_overlap:
                            # case 4
                            largest_partial_tp_overlap = tp_overlap
                            corresponding_partial_fp_overlap = fp_overlap
                            best_partial_entity = predicted_entity

                # increase the true positive scores of partial_type and partial by the largest overlap found
                statistics[gt_ent_type]["partial_type"]["tp"] += largest_partial_type_tp_overlap
                statistics[gt_ent_type]["partial"]["tp"] += largest_partial_tp_overlap

                # increase the false positive scores of partial_type and partial by the corresponding fp overlap of
                # the best true positive relation
                statistics[gt_ent_type]["partial_type"]["fp"] += corresponding_partial_type_fp_overlap
                statistics[gt_ent_type]["partial"]["fp"] += corresponding_partial_fp_overlap

                # calculate by how much to increase the false negative of partial_type and partial
                # this will be the reverse of the true positive score
                statistics[gt_ent_type]["partial_type"]["fn"] += 1 - largest_partial_type_tp_overlap
                statistics[gt_ent_type]["partial"]["fn"] += 1 - largest_partial_tp_overlap

                if best_strict_entity:
                    statistics[gt_ent_type]["strict"]["tp"] += 1
                    best_strict_entity["used_in_strict_metric"] = True
                else:
                    statistics[gt_ent_type]["strict"]["fn"] += 1

                if best_exact_entity:
                    statistics[gt_ent_type]["exact"]["tp"] += 1
                    best_exact_entity["used_in_exact_metric"] = True
                else:
                    statistics[gt_ent_type]["exact"]["fn"] += 1

                if best_partial_type_entity:
                    best_partial_type_entity["used_in_partial_type_metric"] = True

                if best_partial_entity:
                    best_partial_entity["used_in_partial_metric"] = True

            # Remaining FP by looking from predicted entities
            for predicted_entity in pred_sent:
                predicted_entity_type = predicted_entity["type_"]

                if not predicted_entity.get("used_in_strict_metric", False):
                    statistics[predicted_entity_type]["strict"]["fp"] += 1

                if not predicted_entity.get("used_in_exact_metric", False):
                    statistics[predicted_entity_type]["exact"]["fp"] += 1

                if not predicted_entity.get("used_in_partial_type_metric", False):
                    statistics[predicted_entity_type]["partial_type"]["fp"] += 1

                if not predicted_entity.get("used_in_partial_metric", False):
                    statistics[predicted_entity_type]["partial"]["fp"] += 1

        # Compute per entity Precision / Recall / F1 / Support
        for ent_type in statistics.keys():
            for statistic_type in statistics[ent_type].keys():  # strict, exact, type, partial
                if statistic_type != "support":
                    if statistics[ent_type][statistic_type]["tp"] != 0:
                        precision = 100 * statistics[ent_type][statistic_type]["tp"] / \
                                    (statistics[ent_type][statistic_type]["fp"] +
                                     statistics[ent_type][statistic_type]["tp"])
                        recall = 100 * statistics[ent_type][statistic_type]["tp"] / \
                                 (statistics[ent_type][statistic_type]["fn"] + statistics[ent_type][statistic_type]["tp"])
                    else:
                        precision, recall = 0., 0.

                    if not precision + recall == 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.

                    support = statistics[ent_type]["support"]

                    if ent_type not in clf_report:
                        clf_report[ent_type] = {}
                    clf_report[ent_type][statistic_type] = {
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "Support": support
                    }
                    clf_report[ent_type]["Support"] = support

        # Sort clf report descending
        clf_report = dict(sorted(clf_report.items(), key=lambda item: item[1]["Support"], reverse=True))

        # Compute micro & macro F1 Scores
        support_all = sum([statistics[ent_type]["support"] for ent_type in self.entity_types])
        clf_report["micro avg"] = {}
        clf_report["macro avg"] = {}
        for metric_type in ["strict", "exact", "partial_type", "partial"]:
            # micro
            tp = sum([statistics[ent_type][metric_type]["tp"] for ent_type in self.entity_types])
            fp = sum([statistics[ent_type][metric_type]["fp"] for ent_type in self.entity_types])
            fn = sum([statistics[ent_type][metric_type]["fn"] for ent_type in self.entity_types])

            if tp:
                micro_precision = 100 * tp / (tp + fp)
                micro_recall = 100 * tp / (tp + fn)
                micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            else:
                micro_precision, micro_recall, micro_f1 = 0., 0., 0.

            clf_report["micro avg"][metric_type] = {
                "Precision": micro_precision,
                "Recall": micro_recall,
                "F1": micro_f1,
                "Support": support_all
            }

            # macro
            macro_precision = np.mean([
                clf_report[ent_type][metric_type]["Precision"]
                for ent_type in self.entity_types if clf_report[ent_type]["Support"] > 0
            ])
            macro_recall = np.mean([
                clf_report[ent_type][metric_type]["Recall"]
                for ent_type in self.entity_types if clf_report[ent_type]["Support"] > 0
            ])
            macro_f1 = np.mean([
                clf_report[ent_type][metric_type]["F1"]
                for ent_type in self.entity_types if clf_report[ent_type]["Support"] > 0
            ])

            clf_report["macro avg"][metric_type] = {
                "Precision": macro_precision,
                "Recall": macro_recall,
                "F1": macro_f1,
                "Support": support_all
            }

        if reset:
            self.reset()

        # todo: add ability to choose metric
        return {
            "ner_clf_report": clf_report,
            "ner_micro_f1": clf_report["micro avg"]["strict"],
            "ner_macro_f1": clf_report["macro avg"]["strict"]
        }

    def reset(self):
        self.pred_entities = []
        self.gt_entities = []


class REF1Adjusted(Metric):
    def __init__(self, mode: str = 'strict'):
        super().__init__()
        assert mode in ["strict", "partial_type", "exact", "partial"]
        self.mode = mode

        self.pred_relations: List[List[Dict]] = []
        self.gt_relations: List[List[Dict]] = []
        self.relation_types: Optional[List[str]] = None

    def __call__(self,
                 relations_anno: List[List[Dict]],
                 relations_pred: List[List[Dict]],
                 entities_anno: List[List[Dict]],
                 relation_types: List[str]):
        """Evaluate RE predictions
            Args:
                pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
                gt_relations (list) :    list of list of ground truth relations
                    rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                            "tail": (start_idx (inclusive), end_idx (exclusive)),
                            "head_type": ent_type,
                            "tail_type": ent_type,
                            "type": rel_type}
        """
        if self.relation_types is None:
            self.relation_types = relation_types

        self.pred_relations.extend(relations_pred)

        for b in range(len(relations_anno)):
            rel_sent = []
            for rel in relations_anno[b]:
                head = entities_anno[b][rel["head_idx"]]
                rel["head"] = (head["start"], head["end"])
                rel["head_type"] = head["type_"]

                tail = entities_anno[b][rel["tail_idx"]]
                rel["tail"] = (tail["start"], tail["end"])
                rel["tail_type"] = tail["type_"]

                rel_sent.append(rel)

            self.gt_relations.append(rel_sent)

    @staticmethod
    def calculate_relation_overlap(
            gt_relation_head,
            gt_relation_tail,
            pred_relation_head,
            pred_relation_tail
    ):
        # assume symmetric relation (for now?)
        # todo: maybe add option for non symmetric relation?
        gt_relation_head_span = set(range(min(gt_relation_head), max(gt_relation_head)))
        pred_relation_head_span = set(range(min(pred_relation_head), max(pred_relation_head)))
        gt_relation_tail_span = set(range(min(gt_relation_tail), max(gt_relation_tail)))
        pred_relation_tail_span = set(range(min(pred_relation_tail), max(pred_relation_tail)))

        overlap_between_gt_head_and_pred_head = gt_relation_head_span.intersection(pred_relation_head_span)
        overlap_between_gt_head_and_pred_tail = gt_relation_head_span.intersection(pred_relation_tail_span)

        if len(overlap_between_gt_head_and_pred_head) >= len(overlap_between_gt_head_and_pred_tail):
            gt_head_tp = len(overlap_between_gt_head_and_pred_head) / len(gt_relation_head_span)
            gt_head_fp = \
                len(pred_relation_head_span - overlap_between_gt_head_and_pred_head) / len(pred_relation_head_span)
        else:
            gt_head_tp = len(overlap_between_gt_head_and_pred_tail) / len(gt_relation_head_span)
            gt_head_fp = \
                len(pred_relation_tail_span - overlap_between_gt_head_and_pred_tail) / len(pred_relation_tail_span)

        overlap_between_gt_tail_and_pred_head = gt_relation_tail_span.intersection(pred_relation_head_span)
        overlap_between_gt_tail_and_pred_tail = gt_relation_tail_span.intersection(pred_relation_tail_span)

        if len(overlap_between_gt_tail_and_pred_tail) >= len(overlap_between_gt_tail_and_pred_head):
            gt_tail_tp = len(overlap_between_gt_tail_and_pred_tail) / len(gt_relation_tail_span)
            gt_tail_fp = \
                len(pred_relation_tail_span - overlap_between_gt_tail_and_pred_tail) / len(pred_relation_tail_span)
        else:
            gt_tail_tp = len(overlap_between_gt_tail_and_pred_head) / len(gt_relation_tail_span)
            gt_tail_fp = \
                len(pred_relation_head_span - overlap_between_gt_tail_and_pred_head) / len(pred_relation_head_span)

        return (gt_head_tp + gt_tail_tp) / 2, (gt_head_fp + gt_tail_fp) / 2

    def get_metric(self, reset: bool = False):
        assert len(self.pred_relations) == len(self.gt_relations)

        statistics = {
            rel: {
                "support": 0,
                # Strict: exact boundary surface string match and entity type
                "strict": {"tp": 0, "fp": 0, "fn": 0},
                # Partial & Type: some overlap between the system tagged entity and the gold annotation
                "partial_type": {"tp": 0, "fp": 0, "fn": 0},
                # Exact: exact boundary match over the surface string, regardless of the type
                "exact": {"tp": 0, "fp": 0, "fn": 0},
                # Partial: partial boundary match over the surface string, regardless of the type
                "partial": {"tp": 0, "fp": 0, "fn": 0}
            } for rel in self.relation_types
        }
        clf_report = {}

        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(self.pred_relations, self.gt_relations):
            for rel_type in self.relation_types:

                for ground_truth_relation in gt_sent:
                    ground_truth_head = ground_truth_relation["head"]
                    ground_truth_tail = ground_truth_relation["tail"]
                    ground_truth_head_type = ground_truth_relation["head_type"]
                    ground_truth_tail_type = ground_truth_relation["tail_type"]

                    statistics[rel_type]["support"] += 1

                    # measures how much of the entity was already predicted with the correct type
                    largest_partial_type_tp_overlap = 0
                    corresponding_partial_type_fp_overlap = 0
                    # measures how much of the entity was already predicted regardless of type
                    largest_partial_tp_overlap = 0
                    corresponding_partial_fp_overlap = 0
                    # largest_partial_overlap_fraction = [0, length_ground_truth_entity]

                    best_strict_relation = None
                    best_exact_relation = None
                    best_partial_type_relation = None
                    best_partial_relation = None

                    for predicted_relation in pred_sent:

                        tp_overlap, fp_overlap = self.calculate_relation_overlap(
                            gt_relation_head=ground_truth_head,
                            gt_relation_tail=ground_truth_tail,
                            pred_relation_head=predicted_relation["head"],
                            pred_relation_tail=predicted_relation["tail"]
                        )

                        if (
                                (ground_truth_head_type == predicted_relation["head_type"]
                                 and ground_truth_tail_type == predicted_relation["tail_type"])
                                or
                                (ground_truth_head_type == predicted_relation["tail_type"]
                                 and ground_truth_tail_type == predicted_relation["head_type"])
                        ):
                            # type match

                            if tp_overlap == 1 and fp_overlap == 0:
                                # strict case

                                # save this relation in the temporary best relation variables
                                # in the strict case, they will be the best relation for all 4 types
                                best_strict_relation = predicted_relation
                                best_exact_relation = predicted_relation
                                best_partial_type_relation = predicted_relation
                                best_partial_relation = predicted_relation

                                largest_partial_type_tp_overlap = 1
                                corresponding_partial_type_fp_overlap = 0
                                largest_partial_tp_overlap = 1
                                corresponding_partial_fp_overlap = 0

                                # can break, nothing will be superior
                                break
                            else:
                                if tp_overlap > largest_partial_type_tp_overlap:
                                    # partial_type case
                                    largest_partial_type_tp_overlap = tp_overlap
                                    corresponding_partial_type_fp_overlap = fp_overlap
                                    best_partial_type_relation = predicted_relation
                                if tp_overlap > largest_partial_tp_overlap:
                                    # partial case
                                    largest_partial_tp_overlap = tp_overlap
                                    corresponding_partial_fp_overlap = fp_overlap
                                    best_partial_relation = predicted_relation
                        else:
                            if tp_overlap == 1 and fp_overlap == 0:
                                # exact case
                                largest_partial_tp_overlap = 1
                                corresponding_partial_fp_overlap = 0
                                best_partial_relation = predicted_relation
                                best_exact_relation = predicted_relation
                            elif tp_overlap > largest_partial_tp_overlap:
                                # partial case
                                largest_partial_tp_overlap = tp_overlap
                                corresponding_partial_fp_overlap = fp_overlap
                                best_partial_relation = predicted_relation

                    # increase the true positive scores of partial_type and partial by the largest overlap found
                    statistics[rel_type]["partial_type"]["tp"] += largest_partial_type_tp_overlap
                    statistics[rel_type]["partial"]["tp"] += largest_partial_tp_overlap

                    # increase the false positive scores of partial_type and partial by the corresponding fp overlap of
                    # the best true positive relation
                    statistics[rel_type]["partial_type"]["fp"] += corresponding_partial_type_fp_overlap
                    statistics[rel_type]["partial"]["fp"] += corresponding_partial_fp_overlap

                    # calculate by how much to increase the false negative of partial_type and partial
                    # this will be the reverse of the true positive score
                    statistics[rel_type]["partial_type"]["fn"] += 1 - largest_partial_type_tp_overlap
                    statistics[rel_type]["partial"]["fn"] += 1 - largest_partial_tp_overlap

                    if best_strict_relation:
                        statistics[rel_type]["strict"]["tp"] += 1
                        best_strict_relation["used_in_strict_metric"] = True
                    else:
                        statistics[rel_type]["strict"]["fn"] += 1

                    if best_exact_relation:
                        statistics[rel_type]["exact"]["tp"] += 1
                        best_exact_relation["used_in_exact_metric"] = True
                    else:
                        statistics[rel_type]["exact"]["fn"] += 1

                    if best_partial_type_relation:
                        best_partial_type_relation["used_in_partial_type_metric"] = True

                    if best_partial_relation:
                        best_partial_relation["used_in_partial_metric"] = True

                # loop through all predicted relations again and count false positives by checking if the relation was
                # used in the corresponding metric
                for predicted_relation in pred_sent:
                    if not predicted_relation.get("used_in_strict_metric", False):
                        statistics[rel_type]["strict"]["fp"] += 1
                    if not predicted_relation.get("used_in_partial_type_metric", False):
                        statistics[rel_type]["partial_type"]["fp"] += 1
                    if not predicted_relation.get("used_in_exact_metric", False):
                        statistics[rel_type]["exact"]["fp"] += 1
                    if not predicted_relation.get("used_in_partial_metric", False):
                        statistics[rel_type]["partial"]["fp"] += 1


                # statistics[rel_type]["tp"] += len(pred_rels & gt_rels)
                # statistics[rel_type]["fp"] += len(pred_rels - gt_rels)
                # statistics[rel_type]["fn"] += len(gt_rels - pred_rels)

        # Compute per relation Precision / Recall / F1 / Support
        for rel_type in statistics.keys():
            for statistic_type in statistics[rel_type].keys():  # strict, exact, type, partial
                if statistic_type != "support":
                    if statistics[rel_type][statistic_type]["tp"] != 0:
                        precision = 100 * statistics[rel_type][statistic_type]["tp"] / \
                                    (statistics[rel_type][statistic_type]["fp"] +
                                     statistics[rel_type][statistic_type]["tp"])
                        recall = 100 * statistics[rel_type][statistic_type]["tp"] / \
                                 (statistics[rel_type][statistic_type]["fn"] +
                                  statistics[rel_type][statistic_type]["tp"])
                    else:
                        precision, recall = 0., 0.

                    if not precision + recall == 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.

                    support = statistics[rel_type]["support"]

                    if rel_type not in clf_report:
                        clf_report[rel_type] = {}
                    clf_report[rel_type][statistic_type] = {
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "Support": support
                    }
                    clf_report[rel_type]["Support"] = support

        # Sort clf report descending
        clf_report = dict(sorted(clf_report.items(), key=lambda item: item[1]["Support"], reverse=True))

        # Compute micro & macro F1 Scores
        support_all = sum([statistics[rel_type]["support"] for rel_type in self.relation_types])
        clf_report["micro avg"] = {}
        clf_report["macro avg"] = {}
        micro_f1_to_be_returned = 0
        macro_f1_to_be_returned = 0
        relevant_clf_report = {}
        for metric_type in ["strict", "exact", "partial_type", "partial"]:
            # micro
            tp = sum([statistics[rel_type][metric_type]["tp"] for rel_type in self.relation_types])
            fp = sum([statistics[rel_type][metric_type]["fp"] for rel_type in self.relation_types])
            fn = sum([statistics[rel_type][metric_type]["fn"] for rel_type in self.relation_types])

            if tp:
                micro_precision = 100 * tp / (tp + fp)
                micro_recall = 100 * tp / (tp + fn)
                micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            else:
                micro_precision, micro_recall, micro_f1 = 0., 0., 0.

            clf_report["micro avg"][metric_type] = {
                "Precision": micro_precision,
                "Recall": micro_recall,
                "F1": micro_f1,
                "Support": support_all
            }

            # macro
            macro_precision = np.mean([
                clf_report[rel_type][metric_type]["Precision"]
                for rel_type in self.relation_types if clf_report[rel_type]["Support"] > 0
            ])
            macro_recall = np.mean([
                clf_report[rel_type][metric_type]["Recall"]
                for rel_type in self.relation_types if clf_report[rel_type]["Support"] > 0
            ])
            macro_f1 = np.mean([
                clf_report[rel_type][metric_type]["F1"]
                for rel_type in self.relation_types if clf_report[rel_type]["Support"] > 0
            ])

            clf_report["macro avg"][metric_type] = {
                "Precision": macro_precision,
                "Recall": macro_recall,
                "F1": macro_f1,
                "Support": support_all
            }
            if metric_type == self.mode:
                micro_f1_to_be_returned = micro_f1
                macro_f1_to_be_returned = macro_f1
                relevant_clf_report = {
                    "micro avg": {
                        "Precision": micro_precision,
                        "Recall": micro_recall,
                        "F1": micro_f1,
                        "Support": support_all
                    },
                    "macro avg": {
                        "Precision": macro_precision,
                        "Recall": macro_recall,
                        "F1": macro_f1,
                        "Support": support_all
                    }
                }

        if reset:
            self.reset()

        return {
            "re_clf_report": clf_report,
            "relevant_re_clf_report": relevant_clf_report,
            "re_micro_f1": micro_f1_to_be_returned,
            "re_macro_f1": macro_f1_to_be_returned
        }

    def reset(self):
        self.pred_relations = []
        self.gt_relations = []
