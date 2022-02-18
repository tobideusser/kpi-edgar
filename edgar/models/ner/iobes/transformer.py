import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional

from edgar.trainer.utils import get_device
from edgar.data_classes import Labels

logger = logging.getLogger(__name__)


def get_target_subsequent_mask(max_seq_len: int):
    """get_target_subsequent_mask

    Args:
        max_seq_len (int): maximum sequence length of sentences in batch.

    Returns:
        subsequent_mask (torch.Tensor): subsequent mask of ids input.
    """
    subsequent_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()

    return subsequent_mask


# def get_pad_mask(input_: list):
#     """get_source_pad_mask
#
#     Args:
#         input_ (list): list of words
#
#     Returns:
#         src_mask: source mask
#     """
#     batch_size = len(input_)
#     max_seq_len = max(input_)
#     src_mask = torch.ones(batch_size, max_seq_len).long()
#     for batch_id, num_words in enumerate(input_):
#         src_mask[batch_id, num_words:] = 0
#     return src_mask.unsqueeze(-2).byte()


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self,
                 temperature: torch.Tensor,
                 dropout: float = 0.1):

        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        """Scaled Dot-Product Attention forward
        Args:
            query (torch.Tensor) : the query
            key (torch.Tensor) : the key
            value (torch.Tensor) : the value
            mask (Optional(torch.Tensor)): the mask
        """
        # query: [batch size, num heads, query len, key dim]
        # key: [batch size, num heads, kv len, key dim]
        # value: [batch size, num heads, kv len, value dim]

        # step 1 and 2: matmul and scale
        energy = (query @ key.transpose(-2, -1)) / self.temperature
        # energy: [batch size, num heads, query len, kv len]

        # step 3 (Optional): masked
        batch_size, num_heads, len_q, len_kv = energy.shape

        if mask is not None:
            mask_heads = mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

            # mask_fills works like where mask_heads finds 0 it could be filled by -1e9
            energy = energy.masked_fill(mask_heads == 0, -10000)

        # step 4: softmax
        attention = torch.softmax(energy, dim=-1)
        # attention: [batch size, num heads, query len, kv len]

        # Optional step: check if there are some NaN values.
        nan_mask = torch.isnan(attention)
        num_nan = nan_mask.sum()
        if num_nan > 0:
            logger.warning(f'{num_nan} of {attention.numel()} nan values encountered due to softmax on -inf only. '
                           f'All nan values are replaced by 0.')
            attention = attention.masked_fill(nan_mask, 0.)

        # step 5: matmul with value
        out = self.dropout(attention) @ value
        # out = [batch size, num heads, query len, value dim]

        return out, attention


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention sublayer"""

    def __init__(self,
                 dim_model: int,
                 dim_key: int,
                 dim_value: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 dim_model_encoder: Optional[int] = None):
        super().__init__()

        self.num_heads = num_heads
        self.dim_key = dim_key
        self.dim_value = dim_value

        # if the key and value input tensors have a different model dim than the key input tensor
        # (this happens in the decoder when calculating encoder_src-target multi-head-attention)
        # we instantiate key and value projection weight tensors with the correct input dimensions.

        # if dim_model_encoder_src is not provided we assume all three inputs (key, query and value)
        # have the same model dim
        if dim_model_encoder is None:
            dim_model_encoder = dim_model

        self.query_projection = nn.Linear(in_features=dim_model, out_features=num_heads * dim_key, bias=True)
        self.key_projection = nn.Linear(in_features=dim_model_encoder, out_features=num_heads * dim_key, bias=True)
        self.value_projection = nn.Linear(in_features=dim_model_encoder, out_features=num_heads * dim_value, bias=True)

        temperature = torch.sqrt(torch.FloatTensor([self.dim_key])).to(get_device())

        # scaled dot product attention calculation
        self.attention = ScaledDotProductAttention(temperature=temperature)

        # linear layer applied to the concatenated attention heads
        self.fc = nn.Linear(in_features=num_heads * dim_value, out_features=dim_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """MultiHeadAttention sublayer forward
        Args:
            query (torch.Tensor) : the query
            key (torch.Tensor) : the key
            value (torch.Tensor) : the value
            mask (Optional(torch.Tensor)): the mask
        """
        # self attention: query, key, value = input
        # cross attention: query = input 1 / key, value = input 2

        # query: [batch size, query len, key dim * num heads]
        # key: [batch size, kv len, key dim * num heads]
        # value: [batch size, kv len, value dim * num heads]

        # step 1: projection
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # query: [batch size, query len, key dim * num heads]
        # key: [batch size, kv len, key dim * num heads]
        # value: [batch size, kv len, value dim * num heads]

        query = query.view(*(query.shape[:-1]), self.num_heads, self.dim_key).transpose(-2, -3)
        key = key.view(*(key.shape[:-1]), self.num_heads, self.dim_key).transpose(-2, -3)
        value = value.view(*(value.shape[:-1]), self.num_heads, self.dim_value).transpose(-2, -3)
        # query: [batch size, num heads, query len, key dim]
        # key: [batch size, num heads, kv len, key dim]
        # value: [batch size, num heads, kv len, value dim]

        # step 2: attention layer
        out, attention = self.attention(query, key, value, mask)
        # out = [batch size, num heads, query len, value dim]
        # attention: [batch size, num heads, query len, kv len]

        # step 3: concatenation
        out = out.transpose(-3, -2).contiguous()
        # out = [batch size, query len, num heads, value dim]
        out = out.view(*(out.shape[:-3]), -1, self.num_heads * self.dim_value)
        # out = [batch size, query len, num heads * value dim]

        # step 4: linear transformation
        out = self.fc(out)
        # out = [batch size, query len, model dim]

        return out, attention


class PositionwiseFeedForward(nn.Module):
    """Two feed-forward sublayer"""

    def __init__(self,
                 dim_model: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(in_features=dim_model, out_features=dim_feedforward, bias=True)
        self.fc2 = nn.Linear(in_features=dim_feedforward, out_features=dim_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """Two feed-forward sublayer forward

        Args:
            x (torch.Tensor): a tensor entry
        """
        return self.fc2(self.dropout(torch.nn.functional.relu(self.fc1(x))))


class DecoderLayer(nn.Module):
    """DecoderLayer"""

    def __init__(self,
                 dim_model: int,
                 dim_key: int,
                 dim_value: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float,
                 dim_model_encoder: Optional[int]):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(
            dim_model=dim_model,
            dim_key=dim_key,
            dim_value=dim_value,
            num_heads=num_heads,
            dropout=dropout
        )
        self.self_attention_layer_norm = nn.LayerNorm(dim_model)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.use_cross_attention = True if dim_model_encoder is not None else False
        if self.use_cross_attention:
            self.cross_attention = MultiHeadAttention(
                dim_model=dim_model,
                dim_key=dim_key,
                dim_value=dim_value,
                num_heads=num_heads,
                dropout=dropout,
                dim_model_encoder=dim_model_encoder
            )
            self.cross_attention_layer_norm = nn.LayerNorm(dim_model)
            self.cross_attention_dropout = nn.Dropout(dropout)

        self.positionwise_feedforward = PositionwiseFeedForward(dim_model, dim_feedforward, dropout=dropout)
        self.feedforward_layer_norm = nn.LayerNorm(dim_model)
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(self,
                decoder_input: torch.Tensor,
                encoder_output: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor):
        """ Decoder Layer forward pass

        Args:
           decoder_input (torch.Tensor) : embedded target sequence
           encoder_output (torch.Tensor) : embedded encoder output (e.g. word embeddings from BERT)
           source_mask (torch.Tensor) : mask source (masks padding only)
           target_mask (torch.Tensor) : mask target (masks padding and future values during training)

        """

        decoder_output, decoder_self_attention = self.self_attention(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            mask=target_mask)
        decoder_input = self.self_attention_layer_norm(decoder_input + self.self_attention_dropout(decoder_output))

        decoder_encoder_attention = None
        if self.use_cross_attention:
            decoder_output, decoder_encoder_attention = self.cross_attention(
                query=decoder_input,
                key=encoder_output,
                value=encoder_output,
                mask=source_mask)
            decoder_input = self.cross_attention_layer_norm(decoder_input + self.cross_attention_dropout(decoder_output))

        # positionwise feedforward
        decoder_output = self.positionwise_feedforward(decoder_input)
        decoder_input = self.feedforward_layer_norm(decoder_input + self.feedforward_dropout(decoder_output))

        return decoder_input, decoder_encoder_attention


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int,
                 dropout: float = 0.1,
                 num_positions: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(num_positions, dim_model)
        position = torch.arange(0, num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """ Positional Encoding forward
        Args:
            x (tensor): input embeddings
        """
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """Decoder model"""

    def __init__(self,
                 num_target_vocab: int,
                 dim_model: int = 512,
                 dim_key: int = 64,
                 dim_value: int = 64,
                 num_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 num_positions: int = 5000,
                 dim_model_encoder: Optional[int] = None,
                 word_embedding_dim: Optional[int] = None
                 ):

        super().__init__()

        self.position_encoder = PositionalEncoding(dim_model=dim_model,
                                                   dropout=dropout,
                                                   num_positions=num_positions)

        self.decoder_layer_stack = nn.ModuleList([
            DecoderLayer(dim_model=dim_model,
                         dim_key=dim_key,
                         dim_value=dim_value,
                         num_heads=num_heads,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         dim_model_encoder=dim_model_encoder)
            for _ in range(num_layers)])

        if word_embedding_dim:
            self.fc = nn.Linear(in_features=dim_model + word_embedding_dim, out_features=num_target_vocab, bias=True)
        else:
            self.fc = nn.Linear(in_features=dim_model, out_features=num_target_vocab, bias=True)

    def forward(self,
                encoder_output: torch.Tensor,
                target: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor,
                word_embeddings: Optional[torch.Tensor] = None,
                return_attentions: bool = False):

        """Decoder model forward

        Args:
            encoder_output (tensor): output of the encoder. e.g. bert encoded word embeddings
            target (tensor): embedded target sequence. e.g. ner tags or sentence to generate
            source_mask (tensor): mask of the source.
            target_mask (tensor): mask of the target.
            word_embeddings (tensor): allows for concatentating word embeddings after self attention of label embeddings
            return_attentions (boolean) : if we want to return the attention.
        """

        decoder_encoder_attention_list = []

        decoder_output = self.position_encoder(target)

        for decoder_layer in self.decoder_layer_stack:
            decoder_output, decoder_encoder_attention = decoder_layer(
                decoder_input=decoder_output,
                encoder_output=encoder_output,
                source_mask=source_mask,
                target_mask=target_mask
            )
            decoder_encoder_attention_list += [decoder_encoder_attention] if return_attentions else []

        if word_embeddings is not None:
            decoder_output = torch.cat([word_embeddings, decoder_output], dim=-1)
        logits = self.fc(decoder_output)

        if return_attentions:
            return decoder_output, decoder_encoder_attention_list

        return {'logits': logits,
                'cross_attention': decoder_encoder_attention_list if return_attentions else None}


class NERTransformer(nn.Module):
    """Only contains the Transformer Decoder.
    Handles training vs inference distinction. Prepares inputs, masks, etc."""

    def __init__(self,
                 labels: Labels,
                 input_dim: int,
                 label_embedding_dim: int,
                 dim_key: int = 64,
                 dim_value: int = 64,
                 num_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 num_positions: int = 5000,
                 label_masking: bool = True,
                 add_bos: bool = True,
                 use_cross_attention: bool = False,
                 concat_word_embeddings_after_attention: bool = False
                 ):
        super().__init__()

        self.labels = labels
        self.label_embedding_dim = label_embedding_dim

        self.num_labels = len(self.labels.iobes.idx2val)
        # add additional bos embedding as start embedding for label sequence
        if add_bos:
            self.bos_tag_id = self.num_labels
            num_embeddings = self.num_labels + 1  # we need a [BOS] embedding
            self.label_embeddings = nn.Embedding(num_embeddings, self.label_embedding_dim)
        # use "O" (outside) tag as start embedding
        else:
            self.bos_tag_id = self.labels.iobes.val2idx['O']
            self.label_embeddings = nn.Embedding(self.num_labels, self.label_embedding_dim)
        self.label_masking = label_masking

        self.use_cross_attention = use_cross_attention
        self.concat_word_embeddings_after_attention = concat_word_embeddings_after_attention
        if use_cross_attention or concat_word_embeddings_after_attention:
            model_dim = label_embedding_dim
        else:
            model_dim = label_embedding_dim + input_dim

        self.transformer_decoder = TransformerDecoder(
            num_target_vocab=self.num_labels,
            dim_model=model_dim,
            dim_model_encoder=input_dim if use_cross_attention else None,
            dim_key=dim_key,
            dim_value=dim_value,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_positions,
            dropout=dropout,
            word_embedding_dim=input_dim if concat_word_embeddings_after_attention else None
        )

    def get_label_mask(self, label_ids: torch.Tensor) -> torch.Tensor:

        batch_size = label_ids.shape[0]
        max_seq_len = label_ids.shape[1]
        num_label_types = len(self.labels.iobes.idx2val)

        mask = torch.ones(batch_size, max_seq_len, num_label_types).to(get_device())

        for b in range(batch_size):
            for i in range(max_seq_len):
                try:
                    label_name = self.labels.iobes.idx2val[label_ids[b, i].item()]
                except KeyError:
                    label_name = '[BOS]'

                if label_name.startswith(('[BOS]', 'O', 'E', 'S')):
                    masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
                                  if not value.startswith(('O', 'B', 'S'))]
                else:
                    tag = label_name.split('-')[-1]
                    masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
                                  if value not in [f'I-{tag}', f'E-{tag}']]

                mask[b, i, masked_ids] = 0

        return mask

    def mask_logits(self, logit: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:

        for i in range(logit.shape[0]):
            try:
                label_name = self.labels.iobes.idx2val[label_id[i].item()]
            except KeyError:
                label_name = '[BOS]'

            if label_name.startswith(('[BOS]', 'O', 'E', 'S')):
                masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
                              if not value.startswith(('O', 'B', 'S'))]
            else:
                tag = label_name.split('-')[-1]
                masked_ids = [key for key, value in self.labels.iobes.idx2val.items()
                              if value not in [f'I-{tag}', f'E-{tag}']]

            logit[i, :, masked_ids] = -1e12
        return logit

    def forward(self, word_embeddings: torch.Tensor, label_ids: torch.Tensor, pad_mask: torch.Tensor):
        batch_size = word_embeddings.shape[0]
        seq_len = word_embeddings.shape[1]

        bos_id = torch.zeros(batch_size, 1).long().fill_(self.bos_tag_id).to(get_device())
        bos_embedding = self.label_embeddings(bos_id)

        label_embeddings = self.label_embeddings(label_ids.to(get_device()))
        label_embeddings_shifted = torch.cat([bos_embedding, label_embeddings], dim=1)[:, :-1, :]
        label_ids_shifted = torch.cat([bos_id, label_ids], dim=1)[:, :-1]

        if self.use_cross_attention:
            source_mask = pad_mask.unsqueeze(-2) & torch.zeros(seq_len, seq_len).fill_diagonal_(1).bool().to(get_device())
            target_mask = pad_mask.unsqueeze(-2) & get_target_subsequent_mask(max_seq_len=seq_len).to(get_device())
            output = self.transformer_decoder(encoder_output=word_embeddings,
                                              target=label_embeddings_shifted,
                                              source_mask=source_mask,
                                              target_mask=target_mask,
                                              return_attentions=False)
        else:
            target_mask = pad_mask.unsqueeze(-2) & get_target_subsequent_mask(max_seq_len=seq_len).to(get_device())

            if self.concat_word_embeddings_after_attention:
                output = self.transformer_decoder(encoder_output=None,
                                                  target=label_embeddings_shifted,
                                                  source_mask=None,
                                                  target_mask=target_mask,
                                                  word_embeddings=word_embeddings,
                                                  return_attentions=False)
            else:
                target_embedding = torch.cat([word_embeddings, label_embeddings_shifted], dim=-1)
                output = self.transformer_decoder(encoder_output=None,
                                                  target=target_embedding,
                                                  source_mask=None,
                                                  target_mask=target_mask,
                                                  return_attentions=False)

        if self.label_masking:
            label_mask = self.get_label_mask(label_ids=label_ids_shifted)
            logits = output['logits'].masked_fill(label_mask == 0, -10000)
            output['logits'] = logits

        return output

    def decode(self, word_embeddings: torch.Tensor, pad_mask: torch.Tensor):
        batch_size = word_embeddings.shape[0]
        seq_len = word_embeddings.shape[1]

        bos_id = torch.zeros(batch_size, 1).long().fill_(self.bos_tag_id).to(get_device())
        bos_embedding = self.label_embeddings(bos_id)

        label_embeddings_shifted = bos_embedding
        label_ids_shifted = bos_id

        source_mask = pad_mask.unsqueeze(-2) & torch.zeros(seq_len, seq_len).fill_diagonal_(1).bool().to(get_device())
        target_mask = pad_mask.unsqueeze(-2)  # & get_target_subsequent_mask(max_seq_len=seq_len)

        logits = []

        for i in range(seq_len):
            if self.use_cross_attention:
                output = self.transformer_decoder(encoder_output=word_embeddings,
                                                  target=label_embeddings_shifted,
                                                  source_mask=source_mask[:, :i+1, :],
                                                  target_mask=target_mask[:, :, :i+1],
                                                  return_attentions=False)
            else:
                if self.concat_word_embeddings_after_attention:
                    output = self.transformer_decoder(encoder_output=None,
                                                      target=label_embeddings_shifted,
                                                      source_mask=None,
                                                      target_mask=target_mask[:, :, :i + 1],
                                                      word_embeddings=word_embeddings[:, :i+1, :],
                                                      return_attentions=False)
                else:
                    target_embeddings = torch.cat([word_embeddings[:, :i+1, :], label_embeddings_shifted], dim=-1)
                    output = self.transformer_decoder(encoder_output=None,
                                                      target=target_embeddings,
                                                      source_mask=None,
                                                      target_mask=target_mask[:, :, :i + 1],
                                                      return_attentions=False)

            next_logit = output['logits'][:, -1, :].unsqueeze(1)

            if self.label_masking:
                # mask out impossible entity tags given the previously predicted tag
                prev_label_id = label_ids_shifted[:, i].unsqueeze(1)
                next_logit = self.mask_logits(next_logit, prev_label_id)

            logits.append(next_logit)

            probs = torch.softmax(next_logit, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1)
            label_embeddings_shifted = torch.cat([label_embeddings_shifted,
                                                  self.label_embeddings(pred_label_id)], dim=1)
            label_ids_shifted = torch.cat([label_ids_shifted, pred_label_id], dim=1)

        logits = torch.cat(logits, dim=1)

        return {'logits': logits}
