from typing import Dict

import torch.nn as nn
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edgar.data_classes import Labels
from edgar.models import JointDecoder  # , SentenceEncoder
from edgar.models.encoders import Encoder
from edgar.trainer.utils import get_device


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

        # encoder code
        model_type = encoder_params.pop("encoder_type_")
        if model_type == "sentenceEncoder":
            encoder = Encoder.from_config(encoder_type_=model_type, tokenizer=tokenizer, **encoder_params).to(
                get_device()
            )
            decoder_input_dim = encoder.encoder.config.hidden_size
        else:
            encoder = Encoder.from_config(encoder_type_=model_type, **encoder_params)
            decoder_input_dim = encoder.emb_dim

        decoder = JointDecoder.from_config(labels=labels, input_dim=decoder_input_dim, **decoder_params).to(
            get_device()
        )

        return cls(
            encoder=encoder,
            decoder=decoder,
        )
