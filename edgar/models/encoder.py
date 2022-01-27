import logging
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from kpi_relation_extractor.common.trainer.utils import get_device
from kpi_relation_extractor.kpi_relation_extraction.models.pooling import (PoolingRNNGlobal, PoolingRNNLocal,
                                                                           token2word_embedding)

logger = logging.getLogger(__name__)


class SentenceEncoder(nn.Module):
    def __init__(self,
                 type_: str,
                 tokenizer: PreTrainedTokenizerFast,
                 finetune: bool = True,
                 word_pooling: str = 'max',
                 **kwargs_encoder
                 ):
        super().__init__()

        if type_ != tokenizer.type_:
            raise ValueError(f'Tokenizer "{tokenizer.type_}" is not compatible with model "{type_}".')

        self.tokenizer = tokenizer
        self.encoder = self._init_encoder(type_=type_, **kwargs_encoder)

        self.encoder.resize_token_embeddings(len(tokenizer))

        self.base_model_prefix = self.encoder.base_model_prefix

        self.output_attentions = kwargs_encoder.get('output_attentions', False)
        self.output_hidden_states = kwargs_encoder.get('output_hidden_states', False)

        self.finetune = finetune

        # Word Pooling function to pool subword representations
        assert word_pooling in [None, "max", "sum", "avg", "first", "rnn_global", "rnn_local", "attention"]
        self.word_pooling = word_pooling

        if self.word_pooling == "rnn_global":
            input_dim = self.encoder.config.hidden_size
            hidden_dim = int(input_dim / 2)
            self.pooling_layer = PoolingRNNGlobal(nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True))
        elif self.word_pooling == "rnn_local":
            input_dim = self.encoder.config.hidden_size
            hidden_dim = int(input_dim / 2)
            self.pooling_layer = PoolingRNNLocal(nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True))
        elif self.word_pooling == 'attention':
            input_dim = self.encoder.config.hidden_size
            self.pooling_layer = nn.Linear(input_dim, 1)
        else:
            self.pooling_layer = None

    @staticmethod
    def _init_encoder(type_: str, **kwargs_encoder):
        try:
            encoder_config, model_kwargs = AutoConfig.from_pretrained(type_,
                                                                      **kwargs_encoder,
                                                                      return_unused_kwargs=True)
        except OSError:
            raise NotImplementedError(f'Decoder "{type_}" is not implemented or could not be loaded from Huggingface.')

        if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
            logger.info(f"Initializing {type_} as an encoder model. Cross attention layers are removed from {type_}."
                        )
            encoder_config.is_decoder = False
            encoder_config.add_cross_attention = False

        model_kwargs['config'] = encoder_config

        encoder = AutoModel.from_pretrained(type_, **model_kwargs)
        try:
            encoder = encoder.get_encoder()
        except AttributeError:
            pass

        return encoder

    @staticmethod
    def _make_encoder_attention_mask(pad_idx: int, input_: torch.Tensor):
        # input_ = [batch size, max_seq_len]
        mask = (input_ != pad_idx)
        return mask

    def forward(self, batch: Dict) -> Dict:

        input_ids = batch["token_ids"].to(get_device())
        attention_mask = self._make_encoder_attention_mask(pad_idx=self.tokenizer.pad_token_id, input_=input_ids)

        if self.finetune:
            output = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=None,  # replaces input_ids if input is embedded
                                  output_attentions=self.output_attentions,  # return decoder attention for logging
                                  output_hidden_states=self.output_hidden_states,  # return decoder hidden states
                                  return_dict=True)
        else:
            self.encoder.eval()
            with torch.no_grad():
                output = self.encoder(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      inputs_embeds=None,  # replaces input_ids if input is embedded
                                      output_attentions=self.output_attentions,  # return decoder attention for logging
                                      output_hidden_states=self.output_hidden_states,  # return decoder hidden states
                                      return_dict=True)

        batch["token_embeddings"] = output.last_hidden_state
        if self.word_pooling:
            batch["word_embeddings"] = token2word_embedding(data=batch,
                                                            pooling=self.word_pooling,
                                                            pooling_layer=self.pooling_layer)
        else:
            batch["word_embeddings"] = output.last_hidden_state[:, 1:-1]
        batch["cls_embedding"] = output.last_hidden_state[:, 0]

        return batch
