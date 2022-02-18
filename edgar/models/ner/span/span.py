import logging
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.trainer.utils import get_device, get_padding_mask, set_seeds, argsort
from edgar.models.ner import NERDecoder
from edgar.models.pooling import word2entity_embedding, PoolingRNNLocal

logger = logging.getLogger(__name__)


def get_all_span_ids(sent_len, max_span_len=10):
    """Return all spans (start (inclusive), end (exclusive)) with length <= max_span_len"""
    span_indices = set()
    for i in range(sent_len):
        for k in range(1, max_span_len + 1):
            # (start indice (inclusive), end_indice (exclusive)) in original sentence :
            span_indices.add((i, min(i + k, sent_len)))

    return span_indices


def get_span_labels(spans, entities=None, neg_sampling=0):
    """Return Ground Truth labels along with spans (with possible negative sampling)"""
    positive_spans = []
    positive_ids = set()

    if entities is not None:
        for ent in entities:
            positive_spans.append(((ent["start"], ent["end"]), ent["type_"]))
            positive_ids.add((ent["start"], ent["end"]))

    negative_spans = [s for s in spans if s not in positive_ids]

    if neg_sampling and len(negative_spans) > neg_sampling:
        negative_spans = np.array(negative_spans)
        set_seeds()
        indices = np.random.choice(np.arange(len(negative_spans)), size=neg_sampling, replace=False)
        set_seeds()
        negative_spans = negative_spans[indices]

    negative_spans = [((s[0], s[1]), "None") for s in negative_spans]

    all_spans = sorted(positive_spans + negative_spans)

    return zip(*all_spans)


def filter_overlapping_spans(batch_pred_entities: List[List[Dict]]) -> List[Dict]:
    batch_pred_entities_filtered: List[Dict] = []
    for entities in batch_pred_entities:
        if entities:
            spans = [(ent['start'], ent['end']) for ent in entities]

            inds_sorted = argsort(spans)
            spans_sorted = [spans[i] for i in inds_sorted]
            scores_sorted = [entities[i]['score'] for i in inds_sorted]

            # Group entity spans together if they overlap
            # e.g. [[3, 7], [17, 18], [18, 19], [18, 20], [19, 20]]
            # --> [[3, 7], [17, 18], [[18, 19], [18, 20], [19, 20]]]
            groups = []
            new_group = [(scores_sorted[0], spans_sorted[0])]
            prev_end = spans_sorted[0][1]
            for (start, end), score in zip(spans_sorted[1:], scores_sorted[1:]):
                if start >= prev_end:
                    groups.append(new_group)
                    new_group = [(score, (start, end))]
                else:
                    new_group.append((score, (start, end)))
                prev_end = end
            groups.append(new_group)

            # get spans per group that don't have the highest score -> to be filtered out
            spans_to_delete = [span[1]
                               for group in groups if len(group) > 1
                               for span in group if span != max(group)]

            # get their corresponding indices in the original list
            inds_to_keep = [i for i, span in enumerate(spans) if span not in spans_to_delete]

            entities_filtered = {(ent['start'], ent['end']): ent['type_']
                                 for i, ent in enumerate(entities) if i in inds_to_keep}

            batch_pred_entities_filtered.append(entities_filtered)
        else:
            batch_pred_entities_filtered.append({})

    return batch_pred_entities_filtered


def span2pred(batch, vocab):
    """Convert span predictions into list of entities ({'start', 'end', 'type_'})"""
    batch_size = batch["ner_output"].size(0)
    scores = torch.softmax(batch['ner_logits'], dim=-1).max(dim=-1)[0]
    batch_pred_entities = []

    for b in range(batch_size):
        # pred_entities = {}
        pred_entities = []
        for span, pred, score in zip(batch["span_ids"][b], batch["ner_output"][b], scores[b]):
            if not pred.item() == vocab.entities.val2idx["None"]:
                pred_entities.append({"start": span[0],
                                      "end": span[1],
                                      "type_": vocab.entities.idx2val[pred.item()],
                                      "score": score})
                # pred_entities[(span[0], span[1])] = vocab.entities.idx2val[pred.item()]

        batch_pred_entities.append(pred_entities)

    return batch_pred_entities


class SpanNERDecoder(NERDecoder):
    def __init__(self,
                 input_dim: int,
                 labels: Labels,
                 neg_sampling: int = 100,
                 max_span_len: int = 10,
                 span_len_embedding_dim: int = 25,
                 pooling_fn: str = "max",
                 dropout: float = 0.,
                 chunk_size: int = 1000,
                 use_cls: bool = False,
                 loss_weight: float = 1.,
                 remove_overlapping_spans: bool = True):
        super().__init__()
        self.labels = labels
        self.neg_sampling = neg_sampling
        self.chunk_size = chunk_size
        self.loss_weight = loss_weight
        self.remove_overlapping_spans = remove_overlapping_spans

        self.pooling_fn = pooling_fn

        if pooling_fn == "rnn_local":
            self.pooling_layer = PoolingRNNLocal(nn.GRU(
                input_dim, int(input_dim / 2), bidirectional=True, batch_first=True
            ))
        elif pooling_fn == "attention":
            self.pooling_layer = nn.Linear(input_dim, 1)
        else:
            self.pooling_layer = None

        self.input_dim = input_dim
        self.max_span_len = max_span_len

        # Use CLS
        self.use_cls = use_cls
        if self.use_cls:
            self.input_dim *= 2

        # Span length embeddings
        self.span_len_embedding_dim = span_len_embedding_dim
        if self.span_len_embedding_dim:
            self.span_len_embedder = nn.Embedding(max_span_len, self.span_len_embedding_dim)
            self.input_dim += self.span_len_embedding_dim

        # Linear Layer
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.input_dim, len(self.labels.entities.idx2val))

    def forward(self, batch: Dict):
        word_embeddings = batch['word_embeddings']
        batch_size, _, embedding_dim = word_embeddings.shape

        # Get all spans and their corresponding labels
        all_span_ids = []
        all_labels = []

        for b in range(batch_size):
            span_ids = get_all_span_ids(batch["n_words"][b], self.max_span_len)
            # If training, possible negative sampling
            if self.training:
                span_ids, labels = get_span_labels(span_ids, batch["entities_anno"][b], neg_sampling=self.neg_sampling)
            # Else no negative sampling (GT is only used to compute loss)
            else:
                span_ids, labels = get_span_labels(
                    span_ids,
                    batch["entities_anno"][b],  # if batch["entities_anno"] is not None else None,
                    neg_sampling=False
                )

            all_span_ids.append(span_ids)
            all_labels.append(labels)

        n_spans = [len(spans) for spans in all_span_ids]
        max_n_spans = max(n_spans)

        # Pool word representations in span_representations
        span_representations = torch.zeros((batch_size, max_n_spans, self.input_dim), device=get_device())
        targets = torch.zeros((batch_size, max_n_spans), device=get_device(), dtype=torch.long)

        for b in range(batch_size):
            for i, ((start, end), label) in enumerate(zip(all_span_ids[b], all_labels[b])):
                if not end - start > self.max_span_len:
                    targets[b, i] = self.labels.entities.val2idx[label]

                    span_representation = word2entity_embedding(
                        word_embeddings=batch["word_embeddings"][b, start: end, :],
                        pooling=self.pooling_fn,
                        pooling_layer=self.pooling_layer
                    )

                    # Concat span length embedding
                    if self.span_len_embedding_dim:
                        span_len = torch.tensor(end - start - 1, dtype=torch.long, device=get_device())
                        span_representation = torch.cat([span_representation, self.span_len_embedder(span_len)], -1)

                    # Concat CLS
                    if self.use_cls:
                        span_representation = torch.cat([span_representation, batch['cls_embedding'][b]], -1)

                    span_representations[b, i] = span_representation

        batch["span_pooled"] = span_representations
        batch["n_spans"] = n_spans
        batch["span_ids"] = all_span_ids
        batch["span_tags"] = targets

        # Classify spans in entities
        logits = torch.zeros((batch_size, max_n_spans, len(self.labels.entities.val2idx)), device=get_device())

        for i in range(0, max_n_spans, self.chunk_size):
            logits[:, i:i + self.chunk_size] = self.linear(self.drop(batch["span_pooled"][:, i:i + self.chunk_size]))

        batch["ner_logits"] = logits
        batch["ner_output"] = torch.argmax(logits, dim=-1)

        # TODO: Why only compute when relations are set?
        # if batch['relations_anno'] is not None:
        if any(batch_entities for batch_entities in batch['entities_anno'] if batch_entities):
            batch["ner_loss"] = self.compute_loss(batch)

        # Convert span predictions into list of entities
        entities_pred = span2pred(batch, self.labels)
        batch["ner_score"] = [[float(e["score"]) for e in sentence_entities] for sentence_entities in entities_pred]
        if self.remove_overlapping_spans and not self.training:
            batch["entities_pred"] = filter_overlapping_spans(entities_pred)
        else:
            batch["entities_pred"] = [{(ent['start'], ent['end']): ent['type_']
                                       for i, ent in enumerate(entities)} for entities in entities_pred]

        return batch

    def compute_loss(self, batch: Dict):
        logits = batch["ner_logits"]
        tags = batch["span_tags"].long()

        seqlens = batch["n_spans"]
        loss_mask = get_padding_mask(seqlens, device=get_device())

        loss = nn.CrossEntropyLoss(reduction="none")(logits.view(-1, len(self.labels.entities.val2idx)), tags.view(-1))
        masked_loss = loss * loss_mask.view(-1).float()

        # normalize per tag
        return masked_loss.sum() / loss_mask.sum()
