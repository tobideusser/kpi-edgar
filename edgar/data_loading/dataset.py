from copy import deepcopy
from typing import List

from torch.utils.data import Dataset as TorchDataset

from edgar.data_classes import Sentence


class KPIRelationDataset(TorchDataset):

    def __init__(self,
                 sentences: List[Sentence]):

        self.sentences = deepcopy(sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sample = self.sentences[idx]
        return sample


def main():
    from pickle import load
    from edgar.data_classes import Corpus

    corpus = load(open("/shared_lt/kpi_relation_extractor/experiments/banz/SubWordTokenization/002/corpus_tokenized.p", "rb"))
    corpus = Corpus.from_dict(corpus)
    dataset = KPIRelationDataset(corpus.sentences)


if __name__ == '__main__':
    main()
