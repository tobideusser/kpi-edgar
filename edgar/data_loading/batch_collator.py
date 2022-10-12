from typing import Tuple, Dict, Any

import torch

from edgar.data_classes import Sentence


class BatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Tuple[Sentence, ...]) -> Dict[str, Any]:

        max_words = max([len(sentence.words) for sentence in batch])
        max_tokens = max([len(sentence.tokens) for sentence in batch])

        try:
            entities_anno_iobes_ids = torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.tensor(sentence.entities_anno_iobes_ids),
                        pad=(0, max_words - sentence.n_words),
                        value=self.pad_token_id,
                    )
                    for sentence in batch
                ]
            )
        except RuntimeError:
            entities_anno_iobes_ids = None

        return {
            "unique_id": [sentence.unique_id for sentence in batch],
            "words": [sentence.words for sentence in batch],
            "n_words": [sentence.n_words for sentence in batch],
            "tokens": [sentence.tokens for sentence in batch],
            "entities_anno": [
                [ent.to_dict() for ent in sentence.entities_anno] if sentence.entities_anno else []
                for sentence in batch
            ],
            "relations_anno": [
                [rel.to_dict() for rel in sentence.relations_anno] if sentence.relations_anno else []
                for sentence in batch
            ],
            "word2token_alignment_mask": torch.stack(
                # we have to subtract 1 from sentence.num_tokens to take the sep token of BERT into account
                [
                    torch.nn.functional.pad(
                        torch.tensor(sentence.word2token_alignment_mask),
                        pad=(0, max_tokens - (sentence.num_tokens - 1), 0, max_words - sentence.n_words),
                        value=self.pad_token_id,
                    )
                    for sentence in batch
                ]
            ),
            "token_ids": torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.tensor(sentence.token_ids),
                        pad=(0, max_tokens - sentence.num_tokens),
                        value=self.pad_token_id,
                    )
                    for sentence in batch
                ]
            ),
            "entities_anno_iobes_ids": entities_anno_iobes_ids,
            "word2token_start_ids": torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.tensor(sentence.word2token_start_ids), pad=(0, max_words - sentence.n_words), value=-1
                    )
                    for sentence in batch
                ]
            ),
            "word2token_end_ids": torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.tensor(sentence.word2token_end_ids), pad=(0, max_words - sentence.n_words), value=-1
                    )
                    for sentence in batch
                ]
            ),
        }
