import torch
import torch.nn as nn

from edgar.data_classes import Labels


class Linear(nn.Module):
    def __init__(self, input_dim: int, labels: Labels, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, len(labels.iobes.idx2val))

    def forward(self, word_embeddings: torch.Tensor, label_ids: torch.Tensor, pad_mask):
        logits = self.linear(self.dropout(word_embeddings))
        return {"logits": logits}

    def decode(self, word_embeddings: torch.Tensor, pad_mask):
        logits = self.linear(self.dropout(word_embeddings))
        return {"logits": logits}
