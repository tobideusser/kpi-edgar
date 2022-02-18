import logging
from abc import abstractmethod
from collections import Counter
from importlib import import_module
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.trainer.utils import get_device, get_padding_mask, set_seeds, argsort
from edgar.models.pooling import word2entity_embedding, PoolingRNNLocal


logger = logging.getLogger(__name__)

NER_DECODER: Dict = {"iobes": "edgar.models.decoder.ner.IobesNERDecoder",
                     "span": "edgar.models.decoder.ner.SpanNERDecoder"}


class NERDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: Dict):
        raise NotImplementedError

    @classmethod
    def from_config(cls,
                    type_: str,
                    *args,
                    **kwargs) -> 'NERDecoder':
        try:
            callable_path = NER_DECODER[type_]
            parts = callable_path.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'NER Decoder "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)


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


class ARDecoder(nn.Module):
    """Autoregressive decoder to take previously predicted tags into account for predicting the next tag
    Applicable at the moment only for IOBES NER Tagging.
    """

    def __init__(self,
                 labels: Labels,
                 label_embedding_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 decoding: str):
        super().__init__()

        self.labels = labels
        self.label_embedding_dim = label_embedding_dim
        self.hidden_dim = hidden_dim

        self.label_embeddings = nn.Embedding(len(self.labels.iobes.idx2val), self.label_embedding_dim)
        self.drop = nn.Dropout(dropout)

        self.decoding = decoding
        if self.decoding == 'rnn':
            self.linear = nn.Linear(self.hidden_dim, len(self.labels.iobes.idx2val))
            self.rnn = nn.GRU(label_embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        elif self.decoding == 'ar_linear':
            self.linear = nn.Linear(self.label_embedding_dim + self.hidden_dim, len(self.labels.iobes.idx2val))
        else:
            raise ValueError(f'Decoding type {self.decoding} is not supported.')

    def mask_logits(self, logit: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:

        for i in range(logit.shape[0]):
            label_name = self.labels.iobes.idx2val[label_id[i].item()]

            if label_name.startswith(('O', 'E', 'S')):
                masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
                              if not value.startswith(('O', 'B', 'S'))]
            else:
                tag = label_name.split('-')[-1]
                masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
                              if value not in [f'I-{tag}', f'E-{tag}']]

            logit[i, :, masked_ids] = -1e12
        return logit

    def forward(self, batch: Dict):
        # Pack input sequence, apply RNN and Unpack output
        word_embeddings = batch['rnn_dec_word_embeddings']
        batch_size = word_embeddings.shape[0]

        label_start_id = torch.zeros(batch_size, 1).long().fill_(self.labels.iobes.val2idx['O']).to(get_device())
        label_start_embedding = self.label_embeddings(label_start_id)

        if self.training and "entities_anno_iobes_ids" in batch:
            label_embeddings = self.label_embeddings(batch["entities_anno_iobes_ids"].to(get_device()))
            label_ids = batch["entities_anno_iobes_ids"].to(get_device())
            label_embeddings_shifted = torch.cat([label_start_embedding, label_embeddings], dim=1)[:, :-1, :]
            label_ids_shifted = torch.cat([label_start_id, label_ids], dim=1)[:, :-1]

        else:
            label_embeddings_shifted = label_start_embedding
            label_ids_shifted = label_start_id

        if self.decoding == 'rnn':
            hidden = torch.zeros(1, batch_size, self.hidden_dim).to(get_device())
        seqlens = max(batch['n_words'])

        logits = []

        for i in range(seqlens):
            label_embedding = label_embeddings_shifted[:, i, :].unsqueeze(1)
            word_embedding = word_embeddings[:, i, :].unsqueeze(1)

            input_ = torch.cat([label_embedding, word_embedding], dim=-1)
            if self.decoding == 'rnn':
                set_seeds()
                output, hidden = self.rnn(input_, hidden)
                logit = self.linear(self.drop(output))
                set_seeds()
            elif self.decoding == 'ar_linear':
                set_seeds()
                logit = self.linear(self.drop(input_))
                set_seeds()
            else:
                raise ValueError(f'Decoding type {self.decoding} is not supported.')

            # mask out impossible entity tags given the previously predicted tag
            label_id = label_ids_shifted[:, i].unsqueeze(1)
            logit = self.mask_logits(logit, label_id)

            probs = torch.softmax(logit, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1)

            if not self.training:
                label_embeddings_shifted = torch.cat([label_embeddings_shifted,
                                                      self.label_embeddings(pred_label_id)], dim=1)
                label_ids_shifted = torch.cat([label_ids_shifted, pred_label_id], dim=1)

            logits.append(logit)

        logits = torch.cat(logits, dim=1)

        return logits


class IobesNERDecoder(NERDecoder):
    def __init__(self,
                 input_dim: int,
                 labels: Labels,
                 dropout: float = 0.,
                 use_cls: bool = False,
                 span_len_embedding_dim: int = 25,
                 max_span_len: int = 100,
                 loss_weight: float = 1.,
                 decoding: str = 'rnn',
                 pooling_fn: str = 'max',
                 label_embedding_dim: int = 128):
        super().__init__()
        self.labels = labels
        self.loss_weight = loss_weight

        self.input_dim = input_dim

        self.decoding = decoding
        # Linear Layer
        if self.decoding == 'linear':
            self.dropout = dropout
            self.drop = nn.Dropout(dropout)
            self.linear = nn.Linear(input_dim, len(self.labels.iobes.idx2val))
        # rnn decoding
        elif self.decoding in ['rnn', 'ar_linear']:
            self.ar_decoder = ARDecoder(labels=labels, label_embedding_dim=label_embedding_dim,
                                        hidden_dim=input_dim, dropout=dropout, decoding=self.decoding)
        else:
            raise ValueError(f'Decoding method {self.decoding} is unknown.')

        self.pooling_fn = pooling_fn
        if pooling_fn == "rnn_local":
            self.pooling_layer = PoolingRNNLocal(nn.GRU(
                input_dim, int(input_dim / 2), bidirectional=True, batch_first=True
            ))
        elif pooling_fn == "attention":
            self.pooling_layer = nn.Linear(input_dim, 1)
        else:
            self.pooling_layer = None

        self.use_cls = use_cls
        if self.use_cls:
            self.input_dim *= 2

        # Span length embeddings
        self.max_span_len = max_span_len
        self.span_len_embedding_dim = span_len_embedding_dim
        if self.span_len_embedding_dim:
            self.span_len_embedder = nn.Embedding(max_span_len, self.span_len_embedding_dim)
            self.input_dim += self.span_len_embedding_dim

    def forward(self, batch: Dict):
        word_embeddings = batch['word_embeddings']

        # if self.use_cls:
        #     cls_embeddings = batch['cls_embedding'].unsqueeze(1).repeat(1, word_embeddings.shape[1], 1)
        #     word_embeddings = torch.cat([word_embeddings, cls_embeddings], dim=-1)
        #     batch['use_cls'] = True

        if self.decoding == 'linear':
            # Classify spans in entities with linear layer
            set_seeds()
            logits = self.linear(self.drop(word_embeddings))
            set_seeds()
        elif self.decoding in ['rnn', 'ar_linear']:
            # Classify spans in entities with recurrent layer
            batch['rnn_dec_word_embeddings'] = word_embeddings
            logits = self.ar_decoder(batch)
            del batch['rnn_dec_word_embeddings']
        else:
            raise ValueError(f'Decoding method {self.decoding} is unknown.')

        batch["ner_logits"] = logits
        batch["ner_output"] = torch.argmax(logits, dim=-1)

        # Convert iobes predictions into list of entities
        batch["entities_pred"] = iobes2pred(batch, self.labels)

        batch_ner_scores = []
        for i, sentence in enumerate(batch["entities_pred"]):
            ner_scores = []
            for span, entity in sentence.items():
                start, end = span
                # get most common class index of predicted entity range
                most_common_index = Counter(batch['ner_output'][i, start:end].tolist()).most_common(1)[0][0]

                # calculate entity score as mean of individual word probabilities
                entity_score = torch.softmax(batch["ner_logits"][i, start:end, :],
                                             dim=-1)[:, most_common_index].mean().item()
                ner_scores.append(entity_score)
            batch_ner_scores.append(ner_scores)

        batch["ner_score"] = batch_ner_scores

        # todo: replace dummy ner score with average score over all predicted iobes tags (will be displayed in the
        #       prediction html file)
        # batch["ner_score"] = [[0. for _ in sentence] for sentence in batch["entities_pred"]]

        # Compute span representations
        batch_size = word_embeddings.shape[0]
        all_filtered_span_ids = []
        for b in range(batch_size):
            if self.training:
                entities_anno = {(ent['start'], ent['end']): ent['type_'] for ent in batch["entities_anno"][b]}
                entities_combined = entities_anno
            else:
                entities_combined = batch["entities_pred"][b]

            all_filtered_span_ids.append(list(entities_combined))

        n_spans = [len(s) for s in all_filtered_span_ids]
        max_n_spans = max(n_spans)
        span_representations = torch.zeros((batch_size, max_n_spans, self.input_dim), device=get_device())

        for b in range(batch_size):
            for i, (start, end) in enumerate(all_filtered_span_ids[b]):
                if end - start <= self.max_span_len:
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
        batch["span_ids"] = all_filtered_span_ids

        # TODO: Why only compute when relations are set?
        # if batch['relations_anno'] is not None:
        if any(batch_entities for batch_entities in batch['entities_anno'] if batch_entities):
            batch["ner_loss"] = self.compute_loss(batch)

        return batch

    def compute_loss(self, batch: Dict):
        logits = batch["ner_logits"]
        tags = batch["entities_anno_iobes_ids"].to(get_device())

        seqlens = batch["n_words"]
        loss_mask = get_padding_mask(seqlens, device=get_device())

        loss = nn.CrossEntropyLoss(reduction="none")(logits.view(-1, len(self.labels.iobes.val2idx)), tags.view(-1))
        masked_loss = loss * loss_mask.view(-1).float()

        # normalize per tag
        return masked_loss.sum() / loss_mask.sum()


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


def iobes2iob(iobes):
    """Converts a list of IOBES tags to IOB scheme."""
    convert_dict = {pfx: pfx for pfx in "IOB"}
    convert_dict.update({"S": "B", "E": "I"})
    return [convert_dict[t[0]] + t[1:] if not t == "O" else "O" for t in iobes]


def extract_iob(iob_tags):
    """Convert list of IOB tags into list of entities ({'start', 'end', 'type'})"""
    entities = {}

    tmp_indices = None
    tmp_type = "O"
    for i, t in enumerate(iob_tags):
        if t[0] == "B":
            if tmp_indices is not None:
                entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
            tmp_type = "-".join(t.split("-")[1:])
            tmp_indices = [i]

        elif t[0] == "O":
            if tmp_indices is not None:
                entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
            tmp_type = None
            tmp_indices = None

        elif t[0] == "I":
            if "-".join(t.split("-")[1:]) == tmp_type and i == tmp_indices[-1] + 1:
                tmp_indices += [i]
            else:
                if tmp_indices is not None:
                    entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                    # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
                tmp_type = "-".join(t.split("-")[1:])
                tmp_indices = [i]

    if tmp_indices is not None:
        entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
        # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})

    return entities


def iobes2pred(batch, vocab):
    """Convert iobes predictions into list of entities ({'start', 'end', 'type_'})"""
    pred_iobes_ids = batch["ner_output"]
    pred_iobes = [[vocab.iobes.idx2val[iobes_id] for iobes_id in sample.cpu().numpy()] for sample in pred_iobes_ids]
    pred_iobes = [p[:n_words] for p, n_words in zip(pred_iobes, batch["n_words"])]

    return [extract_iob(iobes2iob(p)) for p in pred_iobes]
