from typing import Dict, Optional, List, Tuple

import torch
from torch import nn

from edgar.data_classes import Labels
from edgar.trainer.utils import get_device, ceildiv
from edgar.models.ner import NERDecoder
from edgar.models.re import REDecoder


class JointDecoder(nn.Module):
    def __init__(self,
                 ner_decoder: NERDecoder,
                 re_decoder: REDecoder,
                 table_to_text_mode: bool = False,
                 loss_weighting_schedule: Optional[str] = None
                 ):
        super().__init__()
        self.current_epoch: Optional[int] = None
        self.num_epochs: Optional[int] = None

        self.ner_decoder = ner_decoder
        self.re_decoder = re_decoder
        self.table_to_text_mode = table_to_text_mode

        self.loss_weighting_schedule = loss_weighting_schedule
        self.ner_weights: Optional[List[float]] = None
        self.re_weights: Optional[List[float]] = None

    def get_loss_weights(self) -> Tuple[float, float]:
        if not self.ner_weights and not self.re_weights:
            try:
                max_value = int(self.loss_weighting_schedule.split('_')[-1])
            except ValueError:
                max_value = 1

            if 'linear' in self.loss_weighting_schedule:
                self.ner_weights = [max_value - max_value * i / (self.num_epochs - 1) for i in range(self.num_epochs)]
                self.re_weights = [max_value * i / (self.num_epochs - 1) for i in range(self.num_epochs)]
            elif 'alternating' in self.loss_weighting_schedule:
                self.ner_weights = ([max_value, 0] * ceildiv(self.num_epochs, 2))[:self.num_epochs]
                self.re_weights = ([0, max_value] * ceildiv(self.num_epochs, 2))[:self.num_epochs]
            else:
                raise NotImplementedError

        return self.ner_weights[self.current_epoch - 1], self.re_weights[self.current_epoch - 1]

    def forward(self, batch: Dict):

        batch = self.ner_decoder(batch)
        batch = self.re_decoder(batch)

        ner_loss = batch.get('ner_loss')
        re_loss = batch.get('re_loss')
        ner_loss = ner_loss if ner_loss is not None else torch.tensor(0.).to(get_device())
        re_loss = re_loss if re_loss is not None else torch.tensor(0.).to(get_device())

        if self.loss_weighting_schedule and self.num_epochs and self.current_epoch:
            self.ner_decoder.loss_weight, self.re_decoder.loss_weight = self.get_loss_weights()

        batch['loss'] = self.ner_decoder.loss_weight * ner_loss + self.re_decoder.loss_weight * re_loss

        return batch

    @classmethod
    def from_config(cls,
                    labels: Labels,
                    input_dim: int,
                    ner_params: Dict,
                    re_params: Dict,
                    table_to_text_mode: bool = False,
                    loss_weighting_schedule: Optional[str] = None) -> 'JointDecoder':

        entity_dim = input_dim * 2 if ner_params.get('use_cls') else input_dim
        entity_dim += ner_params.get('span_len_embedding_dim', 0)
        return cls(
            ner_decoder=NERDecoder.from_config(labels=labels, input_dim=input_dim, **ner_params),
            re_decoder=REDecoder(
                labels=labels,
                entity_dim=entity_dim,
                context_dim=input_dim,
                table_to_text_mode=table_to_text_mode,
                **re_params),
            loss_weighting_schedule=loss_weighting_schedule
        )
