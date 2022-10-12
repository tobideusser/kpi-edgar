from copy import deepcopy
from typing import List

from torch.utils.data import Dataset as TorchDataset

from edgar.data_classes import Sentence


class KPIRelationDataset(TorchDataset):
    def __init__(self, sentences: List[Sentence]):

        self.sentences = deepcopy(sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sample = self.sentences[idx]
        return sample
