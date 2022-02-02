from typing import Dict

import numpy as np
import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.trainer.utils import get_device, pad_mask, set_seeds, argsort
from edgar import (ALLOWED_RELATIONS, ALLOWED_1_TO_N_ENTITIES,
                                                            ALLOWED_N_TO_1_ENTITIES)
from edgar.models.pooling import word2entity_embedding, PoolingRNNLocal


class REDecoder(nn.Module):
    def __init__(self,
                 entity_dim: int,
                 labels: Labels,
                 neg_sampling: int = 100,
                 use_inbetween_context: bool = True,
                 context_dim: int = 0,
                 biaffine: bool = False,
                 dropout: float = 0.0,
                 pooling_fn: str = "max",
                 chunk_size: int = 1000,
                 threshold: float = 0.5,
                 loss_weight: float = 1.,
                 remove_overlapping_relations: bool = True,
                 filter_impossible_relations: bool = True):
        """Linear Scorer : MLP([ent1, ent2])   +  optionally [pooled_middle_context, bilinear(ent1, ent2)]"""
        super().__init__()
        self.entity_dim = entity_dim
        self.input_dim = entity_dim * 2
        self.labels = labels
        self.neg_sampling = neg_sampling
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.loss_weight = loss_weight
        self.filter_impossible_relations = filter_impossible_relations
        self.remove_overlapping_relations = remove_overlapping_relations

        self.use_inbetween_context = use_inbetween_context

        if self.use_inbetween_context:
            self.input_dim += context_dim

        self.pooling_fn = pooling_fn

        if pooling_fn == "rnn_local":
            self.pooling_layer = PoolingRNNLocal(nn.GRU(
                context_dim, int(context_dim / 2), bidirectional=True, batch_first=True
            ))
        elif pooling_fn == "attention":
            self.pooling_layer = nn.Linear(context_dim, 1)
        else:
            self.pooling_layer = None

        # self.context_dim = context_dim
        # # Add context representation of same size than both entities
        # if self.context_dim:
        #     self.context_pool = lambda x: x.max(0)[0]
        #     self.input_dim += self.context_dim

        self.biaffine = biaffine
        # Add bilinear term
        if self.biaffine:
            self.bilinear = nn.Bilinear(self.entity_dim,
                                        self.entity_dim,
                                        len(self.labels.relations.idx2val),
                                        bias=False)

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

        self.linear = nn.Linear(self.input_dim, len(self.labels.relations.idx2val))

    def forward(self, batch: Dict):
        batch_size, max_n_words, context_dim = batch['word_embeddings'].shape

        all_pair_ids = []
        all_labels = []
        all_filtered_span_ids = []

        for b in range(batch_size):

            if self.training:
                entities_anno = {(ent['start'], ent['end']): ent['type_'] for ent in batch["entities_anno"][b]}
                entities_combined = entities_anno
                # entities_combined = {**batch["entities_pred"][b], **entities_anno}
            else:
                entities_combined = batch["entities_pred"][b]

            all_filtered_span_ids.append(list(entities_combined))

            if len(entities_combined) > 1:
                # entity_pair_spans = []
                # TODO: Note i1 < i2 enforces that entity pairs are only predicted in one direction. This is problematic
                #  if we have a dataset where relations are not symmetric
                if self.filter_impossible_relations:
                    entity_pair_spans = [(span1, span2)
                                         if span1 <= span2
                                         else
                                         (span2, span1)
                                         for i1, (span1, type1) in enumerate(entities_combined.items())
                                         for i2, (span2, type2) in enumerate(entities_combined.items())
                                         if i1 < i2 and {type1, type2} in ALLOWED_RELATIONS]
                else:
                    entity_pair_spans = [(span1, span2)
                                         if span1 <= span2
                                         else
                                         (span2, span1)
                                         for i1, (span1, type1) in enumerate(entities_combined.items())
                                         for i2, (span2, type2) in enumerate(entities_combined.items())
                                         if i1 < i2]

            # # Filter entity spans AND SAMPLE
            # gt_spans = [(ent["start"], ent["end"]) for ent in batch["entities_anno"][b]]
            # # pred_spans = [(ent["start"], ent["end"]) for ent in batch["pred_entities"][b]]
            # pred_spans = batch["entities_pred"][b].keys()
            #
            # if self.training:
            #     # Keep GT spans and Predicted spans in training
            #     filtered_spans = list(set(gt_spans).union(set(pred_spans)))
            # else:
            #     # Keep only Predicted spans in inference
            #     filtered_spans = pred_spans
            #
            # all_filtered_span_ids.append(filtered_spans)
            #
            # # Needs at least 2 spans to classify relations
            # if len(filtered_spans) > 1:
            #     # Get all span pairs and labels
            #     filtered_pairs = [(a, b) for a in filtered_spans for b in filtered_spans if not a == b]

                # If training, possible negative sampling
                if self.training:
                    filtered_pairs, filtered_pairs_labels = get_pair_labels(entity_pair_spans,
                                                                            batch["relations_anno"][b],
                                                                            batch["entities_anno"][b],
                                                                            neg_sampling=self.neg_sampling)
                # Else do not pass ground_truth information
                # /!\ This results in an inaccurate loss computation in eval
                # But it enables not to consider GT entities not predicted as entities
                else:
                    filtered_pairs, filtered_pairs_labels = get_pair_labels(entity_pair_spans,
                                                                            neg_sampling=False)

            # Otherwise dummy relation
            else:
                filtered_pairs = [((0, 1), (0, 1))]
                filtered_pairs_labels = ["None"]

            all_pair_ids.append(filtered_pairs)
            all_labels.append(filtered_pairs_labels)

        batch["pair_ids"] = all_pair_ids
        batch["n_pairs"] = [len(p) for p in all_pair_ids]
        max_n_relations = max(batch["n_pairs"])

        # # If IOBES NER : compute span representations (already done in Span NER)
        # if "span_ids" not in batch.keys():
        #     n_spans = [len(s) for s in all_filtered_span_ids]
        #     max_n_spans = max(n_spans)
        #     span_representations = torch.zeros((batch_size, max_n_spans, context_dim), device=get_device())
        #
        #     for b in range(batch_size):
        #         for i, (start, end) in enumerate(all_filtered_span_ids[b]):
        #             span_representations[b, i] = word2entity_embedding(
        #                 word_embeddings=batch["word_embeddings"][b, start: end, :],
        #                 pooling=self.pooling_fn,
        #                 pooling_layer=self.pooling_layer
        #             )
        #     if 'use_cls' in batch:
        #         cls_embeddings = batch['cls_embedding'].unsqueeze(1).repeat(1, span_representations.shape[1], 1)
        #         span_representations = torch.cat([span_representations, cls_embeddings], dim=-1)
        #     batch["span_pooled"] = span_representations
        #     batch["n_spans"] = n_spans
        #     batch["span_ids"] = all_filtered_span_ids

        # Get span pairs representations and labels
        all_pair_representations = torch.zeros((batch_size, max_n_relations, self.input_dim), device=get_device())
        targets = torch.zeros((batch_size, max_n_relations, len(self.labels.relations.val2idx)), device=get_device())

        for b in range(batch_size):
            # Needs at least 2 spans to classify relations
            if len(all_filtered_span_ids[b]) > 1 and all_labels[b] != ['None']:
                # Dict that maps a span (start, end) to its index in span_ids to retrieve pooled representations
                span2idx = {span: idx for idx, span in enumerate(batch["span_ids"][b])}

                filtered_pairs_representations = []
                filtered_pairs_context = []

                for i, (arg1, arg2) in enumerate(all_pair_ids[b]):
                    # Concat both argument representations for each relation

                    filtered_pairs_representations.append(
                        torch.cat([batch["span_pooled"][b][span2idx[tuple(arg1)]],
                                   batch["span_pooled"][b][span2idx[tuple(arg2)]]], -1))

                    # Get pooled Middle context
                    if self.use_inbetween_context:
                        if tuple(arg1) < tuple(arg2):
                            begin, end = arg1[1], arg2[0]
                        else:
                            begin, end = arg2[1], arg1[0]

                        if end - begin > 0:
                            filtered_pairs_context.append(word2entity_embedding(
                                word_embeddings=batch["word_embeddings"][b, begin: end, :],
                                pooling=self.pooling_fn,
                                pooling_layer=self.pooling_layer
                            ))
                        else:
                            filtered_pairs_context.append(torch.zeros(context_dim, device=get_device()))

                    # Add label to targets if not None
                    if all_labels[b][i] != "None":
                        targets[b, i, self.labels.relations.val2idx[all_labels[b][i]]] = 1

                filtered_pairs_representations = torch.stack(filtered_pairs_representations)

                if self.use_inbetween_context:
                    filtered_pairs_context = torch.stack(filtered_pairs_context)
                    filtered_pairs_representations = torch.cat([filtered_pairs_representations, filtered_pairs_context],
                                                               -1)
                all_pair_representations[b, :len(filtered_pairs_representations), :] = filtered_pairs_representations

        batch["pair_pooled"] = all_pair_representations
        batch["pair_tags"] = targets

        # Classify pairs
        logits = torch.zeros((batch_size, max_n_relations, len(self.labels.relations.val2idx)), device=get_device())
        for i in range(0, max_n_relations, self.chunk_size):
            pair_chunk_rep = batch["pair_pooled"][:, i:i + self.chunk_size]
            set_seeds()
            logits[:, i:i + self.chunk_size] = self.linear(self.drop(pair_chunk_rep))
            set_seeds()

            if self.biaffine:
                head = pair_chunk_rep[:, :, :self.entity_dim].contiguous()
                tail = pair_chunk_rep[:, :, self.entity_dim: 2 * self.entity_dim].contiguous()

                set_seeds()
                logits[:, i:i + self.chunk_size] += self.bilinear(self.drop(head), self.drop(tail))
                set_seeds()

        batch["re_scores"] = nn.Sigmoid()(logits)
        batch["re_output"] = batch["re_scores"] > self.threshold

        # Convert pair predictions into list of relations
        batch["relations_pred"] = pair2pred(batch, self.labels,
                                            remove_overlapping_relations=self.remove_overlapping_relations)
        # TODO: if at least one sample in batch has relation annotations we compute the loss
        # if batch['relations_anno'] is not None:
        if any(batch_relations for batch_relations in batch['relations_anno'] if batch_relations):
            batch["re_loss"] = self.compute_loss(batch)
        batch["relation_types"] = [rel for rel in self.labels.relations.val2idx.keys()]
        batch["entity_types"] = [ent for ent in self.labels.entities.val2idx.keys() if ent != 'None']

        return batch

    @staticmethod
    def compute_loss(batch: Dict):
        scores = batch["re_scores"]
        tags = batch["pair_tags"]

        seqlens = batch["n_pairs"]
        loss_mask = pad_mask(seqlens, device=get_device())

        # loss = nn.BCELoss(reduction="none")(scores.view(-1, len(self.vocab.relations.val2idx)), tags.view(-1))
        loss = nn.BCELoss(reduction="none")(scores, tags)

        masked_loss = loss * loss_mask.float().unsqueeze(-1)

        # normalize per tag
        return masked_loss.sum() / loss_mask.sum()


def pair2pred(batch, vocab, remove_overlapping_relations: bool = True):
    """Convert span pair predictions into list of relations"""
    pred_entities = batch["entities_pred"]
    batch_size = batch["re_output"].size(0)
    batch_pred_relations = []

    for b in range(batch_size):
        pred_relations = []
        for (head, tail), pred, score in zip(batch["pair_ids"][b], batch["re_output"][b], batch["re_scores"][b]):
            for i in pred.nonzero():
                rel = dict()
                if head > tail:
                    head, tail = tail, head
                rel["head"] = head
                rel["tail"] = tail
                rel["type_"] = vocab.relations.idx2val[i.item()]
                rel["score"] = score.item()

                if tuple(head) in pred_entities[b] and tuple(tail) in pred_entities[b]:
                    rel["head_type"] = pred_entities[b][tuple(head)] if tuple(head) in pred_entities[b] else "None"
                    rel["tail_type"] = pred_entities[b][tuple(tail)] if tuple(tail) in pred_entities[b] else "None"
                    pred_relations.append(rel)
        # only required if there is more than one relation
        if remove_overlapping_relations and len(pred_relations) > 1:
            pred_relations = filter_overlapping_relations(pred_relations)

        batch_pred_relations.append(pred_relations)

    return batch_pred_relations


def filter_overlapping_relations(relations_pred):
    relations_to_delete = set()
    for i, rel_1 in enumerate(relations_pred):
        # entity spans of relation 1
        rel_1_spans = [rel_1["head"], rel_1["tail"]]
        for j, rel_2 in enumerate(relations_pred):
            if i < j:
                # entity spans of relation 2
                rel_2_spans = [rel_2["head"], rel_2["tail"]]
                # todo: this should not be needed. no error in KRE with new data
                # if rel_1_spans == rel_2_spans:
                #     # edge case: the same relation was predicted twice, just delete the second one
                #     # todo: investigate how this happened?
                #     relations_to_delete.add(j)
                # else:

                # get any overlapping entity spans between relations (max 1 overlapping entity possible)
                overlapping_span = tuple(set(rel_1_spans) & set(rel_2_spans))
                if overlapping_span:
                    assert len(overlapping_span) == 1
                    overlapping_span = overlapping_span[0]
                    # get entity name of found overlapping span
                    overlapping_entity_name = rel_1["head_type"] if overlapping_span == rel_1["head"] else rel_1["tail_type"]
                    # get entity names of non overlapping entity for relation 1 and 2
                    rel_1_other_entity_name = rel_1["head_type"] if overlapping_span != rel_1["head"] else rel_1["tail_type"]
                    rel_2_other_entity_name = rel_2["head_type"] if overlapping_span != rel_2["head"] else rel_2["tail_type"]
                    # only delete relations if
                    if (
                            overlapping_entity_name not in ALLOWED_1_TO_N_ENTITIES  # e.g. cy, py, etc.
                            or
                            # both non overlapping entities are of same type
                            # and the type is not an allowed "N to 1" entity
                            (rel_1_other_entity_name == rel_2_other_entity_name
                             and rel_1_other_entity_name not in ALLOWED_N_TO_1_ENTITIES)
                    ):
                        rel_1_dist = rel_1["tail"][0] - rel_1["head"][1]
                        rel_2_dist = rel_2["tail"][0] - rel_1["head"][1]
                        # Delete relation with lower score
                        # or (if the score is equal) the one with the longer distance between its entities
                        if (
                                rel_1["score"] > rel_2["score"]
                                or (rel_1["score"] == rel_2["score"] and rel_1_dist < rel_2_dist)
                        ):
                            relations_to_delete.add(j)
                        else:
                            relations_to_delete.add(i)

    # apply the filter to our list of predicted relations
    filtered_relations_pred = [rel for id_, rel in enumerate(relations_pred) if id_ not in relations_to_delete]
    return filtered_relations_pred


def get_pair_labels(filtered_pairs, relations=None, entities=None, neg_sampling=0):
    """Return Ground Truth labels along with pairs (with possible negative sampling)"""
    positive_pairs = []
    positive_ids = set()

    if relations is not None and entities is not None:
        for rel in relations:
            head, tail = rel["head_idx"], rel["tail_idx"]

            head_span = (entities[head]["start"], entities[head]["end"])
            tail_span = (entities[tail]["start"], entities[tail]["end"])

            rel_type = rel["type_"]
            positive_pairs.append(((head_span, tail_span), rel_type))
            positive_ids.add((head_span, tail_span))

    negative_pairs = [s for s in filtered_pairs if s not in positive_ids]

    if neg_sampling and len(negative_pairs) > neg_sampling:
        negative_pairs = np.array(negative_pairs)
        set_seeds()
        indices = np.random.choice(np.arange(len(negative_pairs)), size=neg_sampling, replace=False)
        set_seeds()
        negative_pairs = negative_pairs[indices]

    negative_pairs = [((tuple(s[0]), tuple(s[1])), "None") for s in negative_pairs]

    all_pairs = positive_pairs + negative_pairs
    if all_pairs:
        ids, labels = zip(*all_pairs)
        sorted_indices = argsort(ids)
        ids = tuple(ids[i] for i in sorted_indices)
        labels = tuple(labels[i] for i in sorted_indices)

    else:
        ids = [((0, 1), (0, 1))]
        labels = ["None"]

    return ids, labels
