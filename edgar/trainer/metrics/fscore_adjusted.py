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


class NERF1(Metric):
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

    @staticmethod
    def overlap_relative_span_size(
            base_span: List,
            span_to_compare: List,
            relative_to_spans_to_compare: bool = False
    ):
        """
        Calculates the relative overlap of the base span to a list of spans. The overlap is relative to the base span by
        default and by setting relative_to_spans_to_compare to True it will be relative to span with the highest
        overlap.
        """
        assert len(base_span) == 2

        # create a set / range spanning from the start (base_span[0]) to the end (base_span[1]) of the span
        base_range = set(range(base_span[0], base_span[1]))

        # the length of the denominator by which the the maximum overlap will be divided. By default, i.e. if
        # relative_to_spans_to_compare is set to False, this will be the length of the base span.
        len_denominator = base_span[1] - base_span[0]

        # variable to hold the maximum absolute overlap
        max_overlap = 0

        # loop through all spans that will be compared
        for span_to_compare in span_to_compare:

            # create a set / range spanning from the start to the end of the span that will be compared to the base span
            stc = set(range(span_to_compare[0], span_to_compare[1]))

            # calculate the actual absolute overlap of the base span and the compared span
            len_overlap = len(stc.intersection(base_range))

            # save the absolute overlap if it is largest measured so far
            if len_overlap > max_overlap:
                max_overlap = len_overlap

                # calculate the length of the denominator if the option is set to use the compared span as base for it
                if relative_to_spans_to_compare:
                    len_denominator = max(span_to_compare) - min(span_to_compare)

        # return the relative overlap
        return max_overlap / len_denominator, max_overlap, len_denominator

    def get_metric(self, reset: bool = False):
        assert len(self.pred_entities) == len(self.gt_entities)

        # statistics = {ent: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for ent in self.entity_types}
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
                largest_partial_type_overlap = 0
                largest_partial_type_overlap_fraction = [0, length_ground_truth_entity]
                # measures how much of the entity was already predicted regardless of type
                largest_partial_overlap = 0
                largest_partial_overlap_fraction = [0, length_ground_truth_entity]

                statistics[gt_ent_type]["support"] += 1

                best_strict_entity = None
                best_partial_type_entity = None
                best_exact_entity = None
                best_partial_entity = None

                for predicted_entity in pred_sent:
                    # predicted_entity_span = [predicted_entity["start"], predicted_entity["end"]]
                    predicted_entity_span = set(range(predicted_entity["start"], predicted_entity["end"]))
                    absolute_overlap = len(predicted_entity_span.intersection(ground_truth_entity_span))
                    relative_overlap = absolute_overlap / length_ground_truth_entity
                    # relative_overlap = self.overlap_relative_span_size(
                    #     base_span=gt_ent_span, span_to_compare=[predicted_entity_span]
                    # )

                    # 5 possible cases (as seen in variable statistics definition):
                    #   case 1: exact boundary surface string match and entity type
                    #   case 2: some overlap between the system tagged entity and the gold annotation
                    #   case 3: exact boundary match over the surface string, regardless of the type
                    #   case 4: partial boundary match over the surface string, regardless of the type
                    #   case 5: no match at all
                    if predicted_entity["type_"] == gt_ent_type:
                        if relative_overlap == 1:
                            # case 1
                            statistics[gt_ent_type]["strict"]["tp"] += 1
                            statistics[gt_ent_type]["exact"]["tp"] += 1
                            largest_partial_type_overlap = 1
                            largest_partial_type_overlap_fraction[0] = absolute_overlap
                            largest_partial_overlap = 1
                            largest_partial_overlap_fraction[0] = absolute_overlap
                            # predicted_entity["strict"] = True
                            # ground_truth_entity["strict"] = 1
                            best_strict_entity = predicted_entity
                            break  # can break loop, since all other predicted entities will be inferior
                        else:
                            if relative_overlap > largest_partial_type_overlap:
                                # case 2
                                largest_partial_type_overlap = relative_overlap
                                largest_partial_type_overlap_fraction[0] = absolute_overlap
                                best_partial_type_entity = predicted_entity
                            if relative_overlap > largest_partial_overlap:
                                # case 4
                                largest_partial_overlap = relative_overlap
                                largest_partial_overlap_fraction[0] = absolute_overlap
                                best_partial_entity = predicted_entity
                    else:
                        if relative_overlap == 1:
                            # case 3
                            statistics[gt_ent_type]["exact"]["tp"] += 1
                            largest_partial_overlap = 1
                            largest_partial_overlap_fraction[0] = absolute_overlap
                            # predicted_entity["exact"] = True
                            best_exact_entity = predicted_entity
                            best_partial_entity = predicted_entity
                        elif relative_overlap > largest_partial_overlap:
                            # case 4
                            largest_partial_overlap = relative_overlap
                            largest_partial_overlap_fraction[0] = absolute_overlap
                            best_partial_entity = predicted_entity

                # increase the true positive scores of partial_type and partial by the largest overlap found
                statistics[gt_ent_type]["partial_type"]["tp"] += largest_partial_type_overlap
                statistics[gt_ent_type]["partial"]["tp"] += largest_partial_overlap

                # calculate by how to increase to increase the false negative of partial_type and partial
                # this will be the reverse the true positive score
                statistics[gt_ent_type]["partial_type"]["fn"] += 1 - largest_partial_type_overlap
                statistics[gt_ent_type]["partial"]["fn"] += 1 - largest_partial_overlap

                if best_strict_entity:
                    best_strict_entity["used_in_strict_metric"] = True
                    best_strict_entity["used_in_partial_type_metric"] = True
                    best_strict_entity["used_in_exact_metric"] = True
                    best_strict_entity["used_in_partial_metric"] = True
                    best_strict_entity["partial_type_overlap"] = 1
                    best_strict_entity["partial_overlap"] = 1
                    best_strict_entity["partial_type_overlap_fraction"] = largest_partial_type_overlap_fraction
                    best_strict_entity["partial_overlap_fraction"] = largest_partial_overlap_fraction
                else:
                    # no strict match was found, increase false negative counter by 1
                    statistics[gt_ent_type]["strict"]["fn"] += 1

                    if best_exact_entity:
                        best_exact_entity["used_in_exact_metric"] = True
                    else:
                        # no exact match was found, increase false negative counter by 1
                        statistics[gt_ent_type]["exact"]["fn"] += 1

                    if best_partial_type_entity:
                        best_partial_type_entity["used_in_partial_type_metric"] = True
                        best_partial_type_entity["partial_type_overlap"] = largest_partial_type_overlap
                        best_partial_type_entity["partial_type_overlap_fraction"] = \
                            largest_partial_type_overlap_fraction
                    if best_partial_entity:
                        best_partial_entity["used_in_partial_metric"] = True
                        best_partial_entity["partial_overlap"] = largest_partial_overlap
                        best_partial_entity["partial_overlap_fraction"] = largest_partial_overlap_fraction

            # FP by looking from predicted entities
            for predicted_entity in pred_sent:
                predicted_entity_type = predicted_entity["type_"]

                if not predicted_entity.get("used_in_strict_metric", False):
                    statistics[predicted_entity_type]["strict"]["fp"] += 1

                if not predicted_entity.get("used_in_exact_metric", False):
                    statistics[predicted_entity_type]["exact"]["fp"] += 1

                if predicted_entity.get("used_in_partial_type_metric", False):
                    len_predicted_entity = predicted_entity["end"] - predicted_entity["start"]
                    statistics[predicted_entity_type]["partial_type"]["fp"] += \
                        (len_predicted_entity - predicted_entity["partial_type_overlap_fraction"][0]) / \
                        predicted_entity["partial_type_overlap_fraction"][1]
                else:
                    statistics[predicted_entity_type]["partial_type"]["fp"] += 1

                if predicted_entity.get("used_in_partial_metric", False):
                    len_predicted_entity = predicted_entity["end"] - predicted_entity["start"]
                    statistics[predicted_entity_type]["partial"]["fp"] += \
                        (len_predicted_entity - predicted_entity["partial_overlap_fraction"][0]) / \
                        predicted_entity["partial_overlap_fraction"][1]
                else:
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


class REF1(Metric):
    def __init__(self, mode: str = 'strict'):
        super().__init__()
        assert mode in ["strict", "boundaries"]
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

    def get_metric(self, reset: bool = False):
        assert len(self.pred_relations) == len(self.gt_relations)

        statistics = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in self.relation_types}
        clf_report = {}

        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(self.pred_relations, self.gt_relations):
            for rel_type in self.relation_types:
                # strict mode takes argument types into account
                if self.mode == "strict":
                    pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                                 rel["type_"] == rel_type
                                 # and
                                 # rel['head_type'] not in ['davon_increase', 'davon_decrease', 'increase_py', 'decrease_py', 'py1']
                                 # and
                                 # rel['tail_type'] not in ['davon_increase', 'davon_decrease', 'increase_py', 'decrease_py', 'py1']
                                 }
                    gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                               rel["type_"] == rel_type
                               # and
                               # rel['head_type'] not in ['davon_increase', 'davon_decrease', 'increase_py', 'decrease_py', 'py1']
                               # and
                               # rel['tail_type'] not in ['davon_increase', 'davon_decrease', 'increase_py', 'decrease_py', 'py1']
                               }

                # boundaries mode only takes argument spans into account
                elif self.mode == "boundaries":
                    pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type_"] == rel_type}
                    gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type_"] == rel_type}

                statistics[rel_type]["tp"] += len(pred_rels & gt_rels)
                statistics[rel_type]["fp"] += len(pred_rels - gt_rels)
                statistics[rel_type]["fn"] += len(gt_rels - pred_rels)

        # Compute per entity Precision / Recall / F1
        for rel_type in statistics.keys():
            if statistics[rel_type]["tp"]:
                precision = 100 * statistics[rel_type]["tp"] / (statistics[rel_type]["fp"] + statistics[rel_type]["tp"])
                recall = 100 * statistics[rel_type]["tp"] / (statistics[rel_type]["fn"] + statistics[rel_type]["tp"])
            else:
                precision, recall = 0., 0.

            if not precision + recall == 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.
            support = statistics[rel_type]["tp"] + statistics[rel_type]["fn"]
            clf_report[rel_type] = {'Precision': precision,
                                    'Recall': recall,
                                    'F1': f1,
                                    'Support': support}

        # Sort clf report descending
        clf_report = dict(sorted(clf_report.items(), key=lambda item: item[1]['Support'], reverse=True))

        # Compute micro F1 Scores
        all_tp = sum([statistics[rel_type]["tp"] for rel_type in self.relation_types])
        all_fp = sum([statistics[rel_type]["fp"] for rel_type in self.relation_types])
        all_fn = sum([statistics[rel_type]["fn"] for rel_type in self.relation_types])

        if all_tp:
            micro_precision = 100 * all_tp / (all_tp + all_fp)
            micro_recall = 100 * all_tp / (all_tp + all_fn)
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        else:
            micro_precision, micro_recall, micro_f1 = 0., 0., 0.
        support_all = all_tp + all_fn

        clf_report['micro avg'] = {'Precision': micro_precision,
                                   'Recall': micro_recall,
                                   'F1': micro_f1,
                                   'Support': support_all}

        # Compute Macro F1 Scores
        macro_precision = np.mean([clf_report[rel_type]["Precision"] for rel_type in self.relation_types])
        macro_recall = np.mean([clf_report[rel_type]["Recall"] for rel_type in self.relation_types])
        macro_f1 = np.mean([clf_report[rel_type]["F1"] for rel_type in self.relation_types])

        clf_report['macro avg'] = {'Precision': macro_precision,
                                   'Recall': macro_recall,
                                   'F1': macro_f1,
                                   'Support': support_all}

        if reset:
            self.reset()

        return {'re_clf_report': clf_report,
                're_micro_f1': micro_f1,
                're_macro_f1': macro_f1}

    def reset(self):
        self.pred_relations = []
        self.gt_relations = []
