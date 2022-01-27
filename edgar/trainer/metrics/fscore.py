from typing import List, Optional, Dict

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

    def get_metric(self, reset: bool = False):
        assert len(self.pred_entities) == len(self.gt_entities)

        statistics = {ent: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for ent in self.entity_types}
        clf_report = {}

        # Count GT entities and Predicted entities
        # n_sents = len(self.gt_entities)
        # n_phrases = sum([len([ent for ent in sent]) for sent in self.gt_entities])
        # n_found = sum([len([ent for ent in sent]) for sent in self.pred_entities])

        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(self.pred_entities, self.gt_entities):
            for ent_type in self.entity_types:
                # if ent_type not in ['davon_increase', 'davon_decrease', 'increase_py', 'decrease_py', 'py1']:
                pred_ents = {(ent["start"], ent["end"]) for ent in pred_sent if ent["type_"] == ent_type}
                gt_ents = {(ent["start"], ent["end"]) for ent in gt_sent if ent["type_"] == ent_type}
                statistics[ent_type]["support"] += len(gt_ents)
                statistics[ent_type]["tp"] += len(pred_ents & gt_ents)
                statistics[ent_type]["fp"] += len(pred_ents - gt_ents)
                statistics[ent_type]["fn"] += len(gt_ents - pred_ents)

        # Compute per entity Precision / Recall / F1 / Support
        for ent_type in statistics.keys():
            if statistics[ent_type]["tp"]:
                precision = 100 * statistics[ent_type]["tp"] / (statistics[ent_type]["fp"] + statistics[ent_type]["tp"])
                recall = 100 * statistics[ent_type]["tp"] / (statistics[ent_type]["fn"] + statistics[ent_type]["tp"])
            else:
                precision, recall = 0., 0.

            if not precision + recall == 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.

            support = statistics[ent_type]["support"]
            clf_report[ent_type] = {'Precision': precision,
                                    'Recall': recall,
                                    'F1': f1,
                                    'Support': support}

        # Sort clf report descending
        clf_report = dict(sorted(clf_report.items(), key=lambda item: item[1]['Support'], reverse=True))

        # Compute micro F1 Scores
        tp_all = sum([statistics[ent_type]["tp"] for ent_type in self.entity_types])
        fp_all = sum([statistics[ent_type]["fp"] for ent_type in self.entity_types])
        fn_all = sum([statistics[ent_type]["fn"] for ent_type in self.entity_types])
        support_all = sum([statistics[ent_type]["support"] for ent_type in self.entity_types])

        if tp_all:
            micro_precision = 100 * tp_all / (tp_all + fp_all)
            micro_recall = 100 * tp_all / (tp_all + fn_all)
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        else:
            micro_precision, micro_recall, micro_f1 = 0., 0., 0.

        clf_report['micro avg'] = {'Precision': micro_precision,
                                   'Recall': micro_recall,
                                   'F1': micro_f1,
                                   'Support': support_all}

        # Compute Macro F1 Scores
        macro_precision = np.mean([clf_report[ent_type]["Precision"] for ent_type in self.entity_types
                                   if clf_report[ent_type]["Support"] > 0])
        macro_recall = np.mean([clf_report[ent_type]["Recall"] for ent_type in self.entity_types
                                if clf_report[ent_type]["Support"] > 0])
        macro_f1 = np.mean([clf_report[ent_type]["F1"] for ent_type in self.entity_types
                            if clf_report[ent_type]["Support"] > 0])

        clf_report['macro avg'] = {'Precision': macro_precision,
                                   'Recall': macro_recall,
                                   'F1': macro_f1,
                                   'Support': support_all}

        if reset:
            self.reset()

        return {'ner_clf_report': clf_report,
                'ner_micro_f1': micro_f1,
                'ner_macro_f1': macro_f1}

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
