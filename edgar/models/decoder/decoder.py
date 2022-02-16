from typing import Dict

import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.trainer.utils import get_device
from edgar.models.decoder import REDecoder, NERDecoder


class JointDecoder(nn.Module):
    def __init__(self,
                 ner_decoder: NERDecoder,
                 re_decoder: REDecoder,
                 ):
        super().__init__()
        self.ner_decoder = ner_decoder
        self.re_decoder = re_decoder

    def forward(self, batch: Dict):

        batch = self.ner_decoder(batch)
        batch = self.re_decoder(batch)

        ner_loss = batch.get('ner_loss')
        re_loss = batch.get('re_loss')
        ner_loss = ner_loss if ner_loss is not None else torch.tensor(0.).to(get_device())
        re_loss = re_loss if re_loss is not None else torch.tensor(0.).to(get_device())
        batch['loss'] = self.ner_decoder.loss_weight * ner_loss + self.re_decoder.loss_weight * re_loss

        return batch

    @classmethod
    def from_config(cls,
                    labels: Labels,
                    input_dim: int,
                    ner_params: Dict,
                    re_params: Dict) -> 'JointDecoder':

        entity_dim = input_dim * 2 if ner_params.get('use_cls') else input_dim
        entity_dim += ner_params.get('span_len_embedding_dim', 0)
        return cls(ner_decoder=NERDecoder.from_config(labels=labels, input_dim=input_dim, **ner_params),
                   re_decoder=REDecoder(labels=labels, entity_dim=entity_dim, context_dim=input_dim, **re_params))
