from typing import Dict

import torch.nn as nn
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edgar.data_classes import Labels
from edgar.trainer.utils import get_device
from edgar.models import SentenceEncoder, JointDecoder


class JointNERAndREModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: Dict) -> Dict:

        batch = self.encoder(batch)
        batch = self.decoder(batch)
        return batch

    def predict(self, batch: Dict) -> Dict:
        return self.forward(batch)

    @classmethod
    def from_config(
            cls,
            encoder_params: Dict,
            decoder_params: Dict,
            tokenizer: PreTrainedTokenizerFast,
            labels: Labels,
    ):

        encoder = SentenceEncoder(tokenizer=tokenizer, **encoder_params).to(get_device())
        decoder_input_dim = encoder.encoder.config.hidden_size
        decoder = JointDecoder.from_config(labels=labels,
                                           input_dim=decoder_input_dim,
                                           **decoder_params).to(get_device())

        return cls(
            encoder=encoder,
            decoder=decoder,
        )
