import logging
import pickle
from typing import Dict

import numpy as np
import torch

from edgar.models.encoders import Encoder
from edgar.trainer.utils import get_device

logger = logging.getLogger(__name__)


class TfIdfEncoder(Encoder):
    def __init__(
        self,
        path_embedding: str,
        embedding_dim: int = 200,
        oov_vector: str = "random",  # how should the out of vocabulary(oov) word should be handled
        seed: int = 100,
        word_pooling: str = None,
    ):
        super().__init__()

        self.emb_dim = embedding_dim

        with open(path_embedding, "rb") as handle:
            self.vectorizer = pickle.load(handle)

    def forward(self, batch: Dict) -> Dict:

        max_word_batch = max(batch["n_words"])
        batch_word_embedding_list = []
        # getting embeddings of each word from tfidf embeddings
        for word_list in batch["words"]:
            len_word_list = len(word_list)
            word_embedding_list = []
            raw_word_store_list = []
            # getting whole sentece embeddings because TfIdf is w.r.t a document
            for word in word_list:
                word_value = word.value.lower()
                raw_word_store_list.append(word_value)
            whole_sentence_emb = self.vectorizer.transform([raw_word_store_list]).toarray()
            # now getting word embeddings
            for word in raw_word_store_list:
                # first getting word_index (similar to bag_of_word (bow))
                word_bow = self.vectorizer.transform([[word]]).toarray()
                word_emb = torch.from_numpy(np.copy(whole_sentence_emb * word_bow))
                word_embedding_list.append(word_emb.reshape((1, -1)))
            # creating a tensor of zeros to make all word embedding vectors of equal batch length
            num_rows = max_word_batch - len_word_list
            if num_rows == 0:
                pass
            else:
                word_embedding_list.append(torch.zeros(num_rows, self.emb_dim))

            # combining the word embedding and appending them into batch embedding list
            batch_word_embedding_list.append(torch.cat(word_embedding_list, dim=0))
        # storing embeddings in batch dict
        batch["word_embeddings"] = torch.stack(batch_word_embedding_list, dim=0).type(torch.float)
        batch["word_embeddings"] = batch["word_embeddings"].to(get_device())
        batch["token_embeddings"] = None
        batch["cls_embedding"] = None
        return batch
