import logging
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from edgar.trainer.utils import get_device


logger = logging.getLogger(__name__)


class PoolingRNNGlobal(nn.Module):
    """Wrapper for torch.nn.RNN to feed unordered and unpacked inputs"""

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.batch_first = self.rnn.batch_first

    def forward(self, batch: Dict, hidden=None):
        # Pack input sequence, apply RNN and Unpack output
        inputs = batch['token_embeddings']
        seqlens = [len(tokens) for tokens in batch['tokens']]
        num_directions = 2 if self.rnn.bidirectional else 1

        packed_inputs = pack_padded_sequence(inputs, seqlens, batch_first=self.batch_first,
                                             enforce_sorted=False)

        self.rnn.flatten_parameters()
        if hidden is None:
            packed_output, hidden = self.rnn(packed_inputs)
        else:
            packed_output, hidden = self.rnn(packed_inputs, hidden)

        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        output = output.view(output.shape[0], output.shape[1], num_directions, self.rnn.hidden_size)

        # create pooled output by concatenating start of word embedding from backward rnn
        #  with end of word embedding from forward rnn
        batched_start_ids = batch["word2token_start_ids"]
        batched_end_ids = batch["word2token_end_ids"]
        output_pooled = torch.zeros(inputs.shape[0],
                                    batched_start_ids.shape[-1],
                                    inputs.shape[-1]).to(get_device())
        for b in range(batched_start_ids.shape[0]):
            for i in range(batched_start_ids.shape[1]):
                backward_rnn_id = batched_start_ids[b, i]
                if backward_rnn_id == -1:
                    break
                forward_rnn_id = batched_end_ids[b, i]

                forward_embedding = output[b, forward_rnn_id, 0]
                backward_embedding = output[b, backward_rnn_id, 1]
                output_pooled[b, i, :] = torch.cat([forward_embedding, backward_embedding], dim=-1)

        return output_pooled


class PoolingRNNLocal(nn.Module):
    """Wrapper for torch.nn.RNN to feed unordered and unpacked inputs"""

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.batch_first = self.rnn.batch_first

    def forward(self, inputs: torch.Tensor, seq_lens: List[int]):
        # Pack input sequence, apply RNN and Unpack output
        num_directions = 2 if self.rnn.bidirectional else 1

        packed_inputs = pack_padded_sequence(inputs, seq_lens, batch_first=self.batch_first,
                                             enforce_sorted=False)

        self.rnn.flatten_parameters()
        packed_output, hidden = self.rnn(packed_inputs)

        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        output = output.view(output.shape[0], output.shape[1], num_directions, self.rnn.hidden_size)

        # create pooled output by concatenating start of word embedding from backward rnn
        #  with end of word embedding from forward rnn
        output_pooled = torch.zeros(inputs.shape[0],
                                    inputs.shape[-1]).to(get_device())
        for b, end_idx in zip(range(inputs.shape[0]), seq_lens):
            forward_embedding = output[b, end_idx - 1, 0]
            backward_embedding = output[b, 0, 1]
            output_pooled[b, :] = torch.cat([forward_embedding, backward_embedding], dim=-1)

        return output_pooled


def token2word_embedding(data,
                         pooling="max",
                         pooling_layer: Optional[Union[PoolingRNNGlobal, PoolingRNNLocal, nn.Linear]] = None):
    """Pool subword bert embeddings into word embeddings"""
    assert pooling in ["first", "max", "sum", "avg", "rnn_global", "rnn_local", "attention"]

    if pooling == "first":
        # embeddings (bs, max_n_tokens, h_dim)

        embeddings = data["token_embeddings"]
        indices = data["word2token_start_ids"].long().to(get_device())

        # set padded indices to max seq len idx
        indices[indices == -1] = embeddings.shape[1] - 1
        indices = indices.unsqueeze(-1).repeat(1, 1, embeddings.size(-1))
        return embeddings.gather(1, indices)

    elif pooling == "rnn_global":
        embeddings = pooling_layer(data)
        return embeddings

    else:
        # embeddings (bs, max_n_tokens, h_dim)
        embeddings = data["token_embeddings"]
        # mask (bs, max_n_words, max_n_tokens)
        mask = data["word2token_alignment_mask"].to(get_device())
        # embeddings (bs, max_n_tokens, h_dim) -> (bs, max_n_words, max_n_tokens, h_dim)
        embeddings = embeddings.unsqueeze(1).repeat(1, mask.size(1), 1, 1)

        if pooling == "max":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), -1e30)
            return embeddings.max(2)[0]

        elif pooling == "rnn_local":
            embeddings_masked = embeddings.masked_fill((mask == 0).unsqueeze(-1), 0.)
            # batch_size x num_word_embeddings x num_token_embeddings x model_dim

            max_num_tokens_per_word: int = (embeddings_masked.sum(dim=-1) != 0.).sum(dim=-1).max().item()
            formatted_embeddings = torch.zeros(embeddings.shape[0],
                                               embeddings.shape[1],
                                               max_num_tokens_per_word,
                                               embeddings.shape[-1]).to(get_device())
            for b in range(embeddings.shape[0]):
                for s in range(embeddings.shape[1]):
                    embeddings_filtered = embeddings[b, s, mask[b, s] == 1, :]
                    formatted_embeddings[b, s, :embeddings_filtered.shape[0], :] = embeddings_filtered
            # batch_size x num_word_embeddings x max_num_tokens_per_word x model_dim

            batch_size, num_word_embeddings = formatted_embeddings.shape[:2]
            formatted_embeddings = formatted_embeddings.view(-1, formatted_embeddings.shape[2], formatted_embeddings.shape[3])
            # (batch_size * num_word_embeddings) x max_num_tokens_per_word x model_dim

            # get num of token embeddings for each word embedding
            seq_lens = formatted_embeddings.count_nonzero(dim=1)[:, 0].tolist()

            # get word embedding ids where pooling is necessary (more than 1 subword embedding)
            pooling_ids = [i for i, elem in enumerate(seq_lens) if elem > 1]
            inputs = formatted_embeddings[pooling_ids, :, :]
            seq_lens = [elem for elem in seq_lens if elem > 1]

            if len(pooling_ids) > 0:
                # pool embeddings via rnn
                rnn_pooled_embeddings = pooling_layer(inputs=inputs, seq_lens=seq_lens)
            # num_of_pooled_embeddings x model_dim

            # create output embeddings
            out_embeddings = torch.zeros(formatted_embeddings.shape[0], formatted_embeddings.shape[-1]).to(get_device())
            for b in range(formatted_embeddings.shape[0]):
                # if id is in pooling ids, insert embedding from rnn pooled embeddings
                if b in pooling_ids:
                    idx = pooling_ids.index(b)
                    out_embeddings[b, :] = rnn_pooled_embeddings[idx, :]
                # else insert first embedding from formatted embeddings
                else:
                    out_embeddings[b, :] = formatted_embeddings[b, 0]

            # change view of out_embeddings
            out_embeddings = out_embeddings.view(batch_size, num_word_embeddings, -1)
            # batch_size x num_word_embeddings x model_dim

            return out_embeddings

        elif pooling == "attention":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), 0)
            logits = pooling_layer(embeddings)
            logits.masked_fill_((mask == 0).unsqueeze(-1), -1e30)

            attention = logits.softmax(dim=2).transpose(-1, -2)
            return (attention @ embeddings).view(embeddings.shape[0], embeddings.shape[1], -1)

        elif pooling == "sum":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), 0)
            return embeddings.sum(2)

        elif pooling == "avg":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), 0)
            non_zero_count = embeddings.count_nonzero(dim=2)
            non_zero_count.masked_fill_(non_zero_count == 0, 1)
            return embeddings.sum(dim=2) / non_zero_count
            # return embeddings.mean(2)


def word2entity_embedding(word_embeddings,
                          pooling="max",
                          pooling_layer: Optional[Union[PoolingRNNGlobal, PoolingRNNLocal, nn.Linear]] = None):
    """Pool a span of word bert embeddings into an entity embedding"""
    assert pooling in ["max", "sum", "avg", "rnn_local", "attention"]

    if pooling == "max":
        return word_embeddings.max(0)[0]

    elif pooling == "rnn_local":

        word_embeddings = word_embeddings.unsqueeze(0)
        # word_embeddings: [batch size x word tokens x model dim]
        seq_len = [word_embeddings.shape[1]]

        # pool embeddings via rnn
        rnn_pooled_embeddings = pooling_layer(inputs=word_embeddings, seq_lens=seq_len)

        return rnn_pooled_embeddings.view(-1)

    elif pooling == "attention":
        logits = pooling_layer(word_embeddings)
        attention = logits.softmax(dim=0)
        return attention.view(-1) @ word_embeddings

    elif pooling == "sum":
        return word_embeddings.sum(0)

    elif pooling == "avg":
        non_zero_count = word_embeddings.count_nonzero(dim=0)
        non_zero_count.masked_fill_(non_zero_count == 0, 1)
        return word_embeddings.sum(dim=0) / non_zero_count
