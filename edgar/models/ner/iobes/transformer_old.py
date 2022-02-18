import logging
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim_model: int,
            dim_key: int,
            dim_value: int,
            num_heads: int,
            dropout: float = 0.,
            dim_model_encoder: Optional[int] = None

    ):
        super().__init__()

        self.dim_value = dim_value
        self.dim_key = dim_key
        self.num_heads = num_heads

        # if the key and value input tensors have a different model dim than the key input tensor
        # (this happens in the decoder when calculating encoder_src-target multi-head-attention)
        # we instantiate key and value projection weight tensors with the correct input dimensions.

        # if dim_model_encoder_src is not provided we assume all three inputs (key, query and value)
        # have the same model dim
        if dim_model_encoder is None:
            dim_model_encoder = dim_model

        self.query_projection = nn.Linear(in_features=dim_model, out_features=dim_key * num_heads, bias=True)
        self.key_projection = nn.Linear(in_features=dim_model_encoder, out_features=dim_key * num_heads, bias=True)
        self.value_projection = nn.Linear(in_features=dim_model_encoder, out_features=dim_value * num_heads, bias=True)

        device = next(self.parameters()).device
        self.scale = torch.sqrt(torch.FloatTensor([self.dim_key])).to(device)

        # linear layer applied to the concatenated attention heads
        self.fc_o = nn.Linear(in_features=num_heads * dim_value, out_features=dim_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # self attention: query, key, value = input
        # cross attention: query = input 1 / key, value = input 2

        # query: [batch size, query len, key dim * num heads]
        # key: [batch size, kv len, key dim * num heads]
        # value: [batch size, kv len, value dim * num heads]

        # project input tensor to query, key and value
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

        energy = (query @ key.transpose(-2, -1)) / self.scale
        # energy: [batch size, num heads, query len, kv len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -10000)

        attention = torch.softmax(energy, dim=-1)
        # attention: [batch size, num heads, query len, kv len]

        nan_mask = torch.isnan(attention)
        num_nan = nan_mask.sum()
        if num_nan > 0:
            logger.warning(f'{num_nan} of {attention.numel()} nan values encountered due to softmax on -inf only. '
                           f'All nan values are replaced by 0.')
            attention = attention.masked_fill(nan_mask, 0.)

        out = self.dropout(attention) @ value
        # out = [batch size, num heads, query len, value dim]

        out = out.transpose(-3, -2).contiguous()
        # out = [batch size, query len, num heads, value dim]

        out = out.view(*(out.shape[:-3]), -1, self.num_heads * self.dim_value)
        # out = [batch size, query len, num heads * value dim]

        out = self.fc_o(out)
        # out = [batch size, query len, model dim]
        return out, attention


class TransformerDecoderLayer(nn.Module):
    """
    A layer holding two attention and a feedforward layer.
    """

    def __init__(
            self,
            dim_model_decoder: int = 512,
            dim_model_encoder: int = 512,
            dim_key: int = 64,
            dim_value: int = 64,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            dim_model=dim_model_decoder,
            dim_key=dim_key,
            dim_value=dim_value,
            num_heads=num_heads,
            dropout=dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(dim_model_decoder)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.multi_head_attention_encoder = MultiHeadAttention(
            dim_model=dim_model_decoder,
            dim_model_encoder=dim_model_encoder,
            dim_key=dim_key,
            dim_value=dim_value,
            num_heads=num_heads,
            dropout=dropout,
            unsqueeze_key_value=True
        )
        self.enc_attn_layer_norm = nn.LayerNorm(dim_model_decoder)
        self.enc_attn_dropout = nn.Dropout(dropout)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(dim_model_decoder,
                                                                     dim_feedforward,
                                                                     dropout)
        self.ff_layer_norm = nn.LayerNorm(dim_model_decoder)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, trg: Tensor, enc_src: Tensor, src_mask: Tensor, trg_mask: Tensor):
        # trg = [batch size, num trg, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, num trg, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.multi_head_attention(query=trg, key=trg, value=trg, mask=trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.self_attn_dropout(_trg))

        # trg = [batch size, num trg, trg len, hid dim]

        # encoder attention
        _trg, attention = self.multi_head_attention_encoder(query=trg, key=enc_src, value=enc_src, mask=src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.enc_attn_dropout(_trg))

        # trg = [batch size, num trg, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.ff_dropout(_trg))

        # trg = [batch size, num trg, trg len, hid dim]
        # attention = [batch size, num trg, n heads, trg len, src len]

        return trg, attention