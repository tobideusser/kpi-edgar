import logging
from collections import Counter
from importlib import import_module
from typing import Dict

import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.models.ner import NERDecoder
from edgar.models.pooling import word2entity_embedding, PoolingRNNLocal
from edgar.trainer.utils import get_device, get_padding_mask

logger = logging.getLogger(__name__)

IOBES_DECODER: Dict = {
    "linear": "edgar.models.ner.iobes.Linear",
    "crf": "edgar.models.ner.iobes.CRF",
    "rnn": "edgar.models.ner.iobes.RNN",
    "transformer": "edgar.models.ner.iobes.NERTransformer",
}


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


class IobesNERDecoder(NERDecoder):
    def __init__(
        self,
        input_dim: int,
        labels: Labels,
        decoding_params: Dict,
        use_cls: bool = False,
        span_len_embedding_dim: int = 25,
        max_span_len: int = 100,
        loss_weight: float = 1.0,
        pooling_fn: str = "max",
        use_ner_hidden_states: bool = False,
    ):
        super().__init__()
        self.labels = labels
        self.loss_weight = loss_weight

        self.input_dim = input_dim

        self.decoder = self.from_config(input_dim=input_dim, labels=labels, **decoding_params)

        # init entity pooling function
        self.pooling_fn = pooling_fn
        if pooling_fn == "rnn_local":
            self.pooling_layer = PoolingRNNLocal(
                nn.GRU(input_dim, int(input_dim / 2), bidirectional=True, batch_first=True)
            )
        elif pooling_fn == "attention":
            self.pooling_layer = nn.Linear(input_dim, 1)
        else:
            self.pooling_layer = None

        self.use_cls = use_cls
        if self.use_cls:
            self.input_dim *= 2

        # init span length embeddings
        self.max_span_len = max_span_len
        self.span_len_embedding_dim = span_len_embedding_dim
        if self.span_len_embedding_dim:
            self.span_len_embedder = nn.Embedding(max_span_len, self.span_len_embedding_dim)
            self.input_dim += self.span_len_embedding_dim

        self.use_ner_hidden_states = use_ner_hidden_states

    def forward(self, batch: Dict):
        word_embeddings = batch["word_embeddings"]
        pad_mask = get_padding_mask(batch["n_words"], device=get_device())

        # Decode label sequence
        if self.training and batch["entities_anno_iobes_ids"] is not None:
            label_ids = batch["entities_anno_iobes_ids"].to(get_device())
            output: Dict = self.decoder(word_embeddings, label_ids, pad_mask)
            logits = output["logits"]

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
                    most_common_index = Counter(batch["ner_output"][i, start:end].tolist()).most_common(1)[0][0]

                    # calculate entity score as mean of individual word probabilities
                    entity_score = (
                        torch.softmax(batch["ner_logits"][i, start:end, :], dim=-1)[:, most_common_index].mean().item()
                    )
                    ner_scores.append(entity_score)
                batch_ner_scores.append(ner_scores)

            batch["ner_score"] = batch_ner_scores
            batch["ner_loss"] = self.compute_loss(batch) if "loss" not in output else output["loss"]

        else:
            output: Dict = self.decoder.decode(word_embeddings, pad_mask)
            logits = output["logits"]

            batch["ner_logits"] = logits
            batch["ner_output"] = (
                torch.argmax(logits, dim=-1) if "best_sequences" not in output else output["best_sequences"]
            )

            # msg = '\n'
            # for seq in batch["ner_output"]:
            #     msg += f"{', '.join([self.labels.iobes.idx2val[label_id] for label_id in seq.tolist()])}\n"
            # msg += '\n'
            # print(msg)

            # Convert iobes predictions into list of entities
            batch["entities_pred"] = iobes2pred(batch, self.labels)

            # annos = [dict(sorted({(entity['start'], entity['end']): entity['type_'] for entity in batch}.items()))
            #          for batch in batch['entities_anno']]
            # print(f'\nAnnos:\n'
            #       f'{annos}\n'
            #       f'Preds:\n'
            #       f'{batch["entities_pred"]}\n\n')

            batch_ner_scores = []
            for i, sentence in enumerate(batch["entities_pred"]):
                ner_scores = []
                for span, entity in sentence.items():
                    start, end = span
                    if "probs" in output:
                        probs = output["probs"][i, start:end]
                    else:
                        probs = torch.softmax(batch["ner_logits"][i, start:end, :], dim=-1).max(dim=-1).values

                    entity_score = probs.mean().item()
                    ner_scores.append(entity_score)
                batch_ner_scores.append(ner_scores)

            batch["ner_score"] = batch_ner_scores

        # Compute span representations
        batch_size = word_embeddings.shape[0]
        all_filtered_span_ids = []
        for b in range(batch_size):
            if self.training:
                entities_anno = {(ent["start"], ent["end"]): ent["type_"] for ent in batch["entities_anno"][b]}
                entities_combined = entities_anno
            else:
                entities_combined = batch["entities_pred"][b]

            all_filtered_span_ids.append(list(entities_combined))

        n_spans = [len(s) for s in all_filtered_span_ids]
        max_n_spans = max(n_spans)
        span_representations = torch.zeros((batch_size, max_n_spans, self.input_dim), device=get_device())

        if "hidden_states" in output and self.use_ner_hidden_states:
            batch["re_embeddings"] = output["hidden_states"]
        else:
            batch["re_embeddings"] = batch["word_embeddings"]

        for b in range(batch_size):
            for i, (start, end) in enumerate(all_filtered_span_ids[b]):
                if end - start <= self.max_span_len:
                    span_representation = word2entity_embedding(
                        word_embeddings=batch["re_embeddings"][b, start:end, :],
                        pooling=self.pooling_fn,
                        pooling_layer=self.pooling_layer,
                    )

                    # Concat span length embedding
                    if self.span_len_embedding_dim:
                        span_len = torch.tensor(end - start - 1, dtype=torch.long, device=get_device())
                        span_representation = torch.cat([span_representation, self.span_len_embedder(span_len)], -1)

                    # Concat CLS
                    if self.use_cls:
                        span_representation = torch.cat([span_representation, batch["cls_embedding"][b]], -1)

                    span_representations[b, i] = span_representation

        batch["span_pooled"] = span_representations
        batch["n_spans"] = n_spans
        batch["span_ids"] = all_filtered_span_ids

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

    @classmethod
    def from_config(cls, type_: str, *args, **kwargs) -> nn.Module:
        try:
            callable_path = IOBES_DECODER[type_]
            parts = callable_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'IOBES Decoder "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)
