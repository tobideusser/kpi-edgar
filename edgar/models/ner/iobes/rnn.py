import logging
from typing import Optional

import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.models.ner.iobes.decoding_utils import expand_inputs
from edgar.trainer.utils import get_device, set_seeds

logger = logging.getLogger(__name__)


class RNN(nn.Module):
    """RNN Decoder to sequentially tag NER tags."""

    def __init__(
        self,
        labels: Labels,
        label_embedding_dim: int,
        input_dim: int,
        dropout: float,
        label_masking: bool = True,
        add_bos: bool = True,
        model: str = "gru",
        combine_inputs: str = "cat",  # 'add'
        num_beams: Optional[int] = 1,
    ):
        super().__init__()

        self.labels = labels

        if combine_inputs == "cat":
            rnn_hidden_dim = label_embedding_dim + input_dim
            self.combine_inputs = torch.cat
            self.label_embedding_dim = label_embedding_dim
        elif combine_inputs == "add":
            self.label_embedding_dim = input_dim
            self.combine_inputs = lambda x, dim: torch.add(x[0], x[1])
            rnn_hidden_dim = input_dim
        else:
            raise NotImplementedError

        self.hidden_dim = input_dim

        self.num_labels = len(self.labels.iobes.idx2val)
        # add additional bos embedding as start embedding for label sequence
        if add_bos:
            self.bos_tag_id = self.num_labels
            num_embeddings = self.num_labels + 1  # we need a [BOS] embedding
            self.label_embeddings = nn.Embedding(num_embeddings, self.label_embedding_dim)
        # use "O" (outside) tag as start embedding
        else:
            self.bos_tag_id = self.labels.iobes.val2idx["O"]
            self.label_embeddings = nn.Embedding(self.num_labels, self.label_embedding_dim)
        self.label_masking = label_masking

        self.drop = nn.Dropout(dropout)

        # linear output layer
        self.linear = nn.Linear(self.hidden_dim, self.num_labels)

        # rnn layer
        self.model_type = model
        if model == "gru":
            self.rnn = nn.GRU(rnn_hidden_dim, input_dim, batch_first=True)
        elif model == "lstm":
            self.rnn = nn.LSTM(rnn_hidden_dim, input_dim, batch_first=True)
        else:
            raise NotImplementedError

        if num_beams and num_beams > 1:
            self.decoding_fn = self.beam_search
        else:
            self.decoding_fn = self.gready_search
        self.num_beams = num_beams
        self.num_beam_hyps_to_keep = 1

    def mask_logits(self, logit: torch.Tensor, label_id: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:

        for i in range(logit.shape[0]):
            try:
                label_name = self.labels.iobes.idx2val[label_id[i].item()]
            except KeyError:
                label_name = "[BOS]"

            if label_name.startswith(("[BOS]", "O", "E", "S")):
                masked_ids = [
                    key for key, value in self.labels.iobes.idx2val.items() if not value.startswith(("O", "B", "S"))
                ]
            else:
                tag = label_name.split("-")[-1]
                masked_ids = [
                    key for key, value in self.labels.iobes.idx2val.items() if value not in [f"I-{tag}", f"E-{tag}"]
                ]

            # # 1: normal token, 0: pad token -> if pad token: predict as O tag
            # if not pad_mask[i]:
            #     masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
            #                   if not value.startswith('O')]

            logit[i, :, masked_ids] = -1e12
        return logit

    def forward(self, word_embeddings: torch.Tensor, label_ids: torch.Tensor, pad_mask: torch.Tensor):
        # Pack input sequence, apply RNN and Unpack output

        batch_size = word_embeddings.shape[0]
        seq_len = word_embeddings.shape[1]

        bos_id = torch.zeros(batch_size, 1).long().fill_(self.bos_tag_id).to(get_device())
        bos_embedding = self.label_embeddings(bos_id)

        label_embeddings = self.label_embeddings(label_ids.to(get_device()))
        label_embeddings_shifted = torch.cat([bos_embedding, label_embeddings], dim=1)[:, :-1, :]
        label_ids_shifted = torch.cat([bos_id, label_ids], dim=1)[:, :-1]

        if self.model_type == "gru":
            hidden = torch.zeros(1, batch_size, self.hidden_dim).to(get_device())
        elif self.model_type == "lstm":
            hidden = (
                torch.zeros(1, batch_size, self.hidden_dim).to(get_device()),
                torch.zeros(1, batch_size, self.hidden_dim).to(get_device()),
            )
        else:
            raise NotImplementedError

        logits = []
        hidden_states = []

        for i in range(seq_len):
            label_embedding = label_embeddings_shifted[:, i, :].unsqueeze(1)
            word_embedding = word_embeddings[:, i, :].unsqueeze(1)

            input_ = self.combine_inputs([label_embedding, word_embedding], dim=-1)
            set_seeds()
            output, hidden = self.rnn(input_, hidden)
            hidden_states.append(output)
            logit = self.linear(self.drop(output))
            set_seeds()

            if self.label_masking:
                # mask out impossible entity tags given the previously predicted tag
                label_id = label_ids_shifted[:, i].unsqueeze(1)
                logit = self.mask_logits(logit, label_id, pad_mask[:, i])

            logits.append(logit)

        logits = torch.cat(logits, dim=1)
        hidden_states = torch.cat(hidden_states, dim=1)

        return {"logits": logits, "hidden_states": hidden_states}

    def decode(self, word_embeddings: torch.Tensor, pad_mask: torch.Tensor):
        return self.decoding_fn(word_embeddings, pad_mask)

    def gready_search(self, word_embeddings: torch.Tensor, pad_mask: torch.Tensor):
        # Greedy Decoding

        batch_size = word_embeddings.shape[0]
        seq_len = word_embeddings.shape[1]

        bos_id = torch.zeros(batch_size, 1).long().fill_(self.bos_tag_id).to(get_device())
        bos_embedding = self.label_embeddings(bos_id)

        label_embeddings_shifted = bos_embedding
        label_ids_shifted = bos_id

        if self.model_type == "gru":
            hidden = torch.zeros(1, batch_size, self.hidden_dim).to(get_device())
        elif self.model_type == "lstm":
            hidden = (
                torch.zeros(1, batch_size, self.hidden_dim).to(get_device()),
                torch.zeros(1, batch_size, self.hidden_dim).to(get_device()),
            )
        else:
            raise NotImplementedError

        logits = []
        hidden_states = []

        for i in range(seq_len):
            label_embedding = label_embeddings_shifted[:, i, :].unsqueeze(1)
            word_embedding = word_embeddings[:, i, :].unsqueeze(1)

            # input_ = torch.cat([label_embedding, word_embedding], dim=-1)
            input_ = self.combine_inputs([label_embedding, word_embedding], dim=-1)
            output, hidden = self.rnn(input_, hidden)
            hidden_states.append(output)
            logit = self.linear(self.drop(output))

            if self.label_masking:
                # mask out impossible entity tags given the previously predicted tag
                label_id = label_ids_shifted[:, i].unsqueeze(1)
                logit = self.mask_logits(logit, label_id, pad_mask[:, i])

            logits.append(logit)

            probs = torch.softmax(logit, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1)
            label_embeddings_shifted = torch.cat(
                [label_embeddings_shifted, self.label_embeddings(pred_label_id)], dim=1
            )
            label_ids_shifted = torch.cat([label_ids_shifted, pred_label_id], dim=1)

        logits = torch.cat(logits, dim=1)
        hidden_states = torch.cat(hidden_states, dim=1)

        return {"logits": logits, "hidden_states": hidden_states}

    def beam_search(self, word_embeddings: torch.Tensor, pad_mask: torch.Tensor):

        batch_size = word_embeddings.shape[0]
        seq_len = word_embeddings.shape[1]

        word_embeddings = expand_inputs(word_embeddings, num_beams=self.num_beams)
        # (batch_size * num_beams, max_seq_len, embedding dim)
        pad_mask = expand_inputs(pad_mask, num_beams=self.num_beams)
        # (batch_size * num_beams, max_seq_len)

        batch_beam_size = batch_size * self.num_beams

        bos_id = torch.zeros(batch_beam_size, 1).long().fill_(self.bos_tag_id).to(get_device())
        bos_embedding = self.label_embeddings(bos_id)

        label_embeddings_shifted = bos_embedding
        label_ids_shifted = bos_id

        if self.model_type == "gru":
            hidden = torch.zeros(1, batch_size, self.hidden_dim).to(get_device())
        elif self.model_type == "lstm":
            hidden = (
                torch.zeros(1, batch_size, self.hidden_dim).to(get_device()),
                torch.zeros(1, batch_size, self.hidden_dim).to(get_device()),
            )
        else:
            raise NotImplementedError

        log_scores = torch.zeros(batch_beam_size, 1).long().fill_(self.bos_tag_id).to(get_device())

        # init attention / hidden states / scores tuples
        scores = ()
        hidden_states = []

        beam_scores = torch.zeros((batch_size, self.num_beams), dtype=torch.float, device=word_embeddings.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * self.num_beams,))

        for i in range(seq_len):

            label_embedding = label_embeddings_shifted[:, i, :].unsqueeze(1)
            word_embedding = word_embeddings[:, i, :].unsqueeze(1)

            # input_ = torch.cat([label_embedding, word_embedding], dim=-1)
            input_ = self.combine_inputs([label_embedding, word_embedding], dim=-1)
            output, hidden = self.rnn(input_, hidden)
            hidden_states.append(output)
            logit = self.linear(self.drop(output))  # (batch_size * num_beams, vocab_size)

            if self.label_masking:
                # mask out impossible entity tags given the previously predicted tag
                label_id = label_ids_shifted[:, i].unsqueeze(1)
                logit = self.mask_logits(logit, label_id, pad_mask[:, i])

            logit = logit.view(batch_beam_size, -1)

            next_log_scores = torch.log_softmax(logit, dim=-1)

            # update beam score only for actual tokens (not padded tokens)
            next_log_scores = next_log_scores.masked_fill(~pad_mask[:, i].view(-1, 1), 0)
            next_scores = next_log_scores + beam_scores[:, None].expand_as(next_log_scores)
            scores += (next_scores,)

            # reshape for beam search
            next_scores = next_scores.view(batch_size, self.num_beams * self.num_labels)

            next_scores, next_tokens = torch.topk(next_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True)
            next_indices = (next_tokens / self.num_labels).long()
            next_tokens = next_tokens % self.num_labels

            next_beam_scores = torch.zeros(
                (batch_size, self.num_beams), dtype=next_scores.dtype, device=word_embeddings.device
            )
            next_beam_tokens = torch.zeros(
                (batch_size, self.num_beams), dtype=next_tokens.dtype, device=word_embeddings.device
            )
            next_beam_indices = torch.zeros(
                (batch_size, self.num_beams), dtype=next_indices.dtype, device=word_embeddings.device
            )

            for batch_idx in range(batch_size):

                # if not pad_mask_not_extended[batch_idx, i]:
                #     # pad the batch
                #     next_beam_scores[batch_idx, :] = 0
                #     next_beam_tokens[batch_idx, :] = 0
                #     next_beam_indices[batch_idx, :] = 0
                #     continue

                beam_idx = 0
                for (next_token, next_score, next_index) in zip(
                    next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx]
                ):
                    batch_beam_idx = batch_idx * self.num_beams + next_index

                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                    # once the beam for next step is full, don't add more tokens to it.
                    if beam_idx == self.num_beams:
                        break
            beam_scores = next_beam_scores.view(-1)

            # input_ids = torch.cat([input_ids[next_beam_indices, :], next_beam_tokens.unsqueeze(-1)], dim=-1)

            # pred_label_id = torch.argmax(probs, dim=-1)
            label_embeddings_shifted = torch.cat(
                [label_embeddings_shifted[next_beam_indices, :], self.label_embeddings(next_beam_tokens.unsqueeze(-1))],
                dim=-2,
            ).view(batch_beam_size, -1, self.label_embedding_dim)
            label_ids_shifted = torch.cat(
                [label_ids_shifted[next_beam_indices, :], next_beam_tokens.unsqueeze(-1)], dim=-1
            ).view(batch_beam_size, -1)

            log_scores = torch.cat(
                [
                    log_scores[next_beam_indices, :].view(batch_beam_size, -1),
                    torch.gather(
                        next_log_scores[next_beam_indices, :].view(batch_beam_size, -1),
                        dim=-1,
                        index=next_beam_tokens.view(-1, 1),
                    ),
                ],
                dim=-1,
            )

        # cut off bos token from sequences and reshape
        label_ids_shifted = label_ids_shifted[:, 1:].view(batch_size, self.num_beams, -1)
        beam_scores = beam_scores.view(batch_size, self.num_beams)
        probs = torch.exp(log_scores)[:, 1:].view(batch_size, self.num_beams, -1)

        # collect beam hypotheses
        beam_hypotheses = []
        for scores, beams, beam_probs in zip(beam_scores, label_ids_shifted, probs):
            beam_hypotheses.append(
                [(score.item(), beam, marginals) for score, beam, marginals in zip(scores, beams, beam_probs)]
            )

        # select the best hypotheses
        best_sequences = []
        best_marginals = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep, device=word_embeddings.device, dtype=torch.float32
        )

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(beam_hypotheses):
            sorted_hyps = sorted(beam_hyp, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_sequence = best_hyp_tuple[1]
                best_prob_sequence = best_hyp_tuple[2]

                # append to lists
                best_sequences.append(best_sequence)
                best_marginals.append(best_prob_sequence)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        best_sequences = torch.stack(best_sequences)
        best_marginals = torch.stack(best_marginals)

        hidden_states = torch.cat(hidden_states, dim=1)

        return {
            "logits": torch.zeros(
                (batch_size, seq_len, self.num_labels), device=word_embeddings.device, dtype=torch.float32
            ),
            "probs": best_marginals,
            "scores": best_scores,
            "best_sequences": best_sequences,
            "hidden_states": hidden_states,
        }
