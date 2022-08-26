import logging
from typing import Dict

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edgar.trainer.utils import get_device
from edgar.models.pooling import (PoolingRNNGlobal, PoolingRNNLocal, token2word_embedding)
from edgar.models.encoders import Encoder


logger = logging.getLogger(__name__)


class EdgarW2VEncoder(Encoder):
    def __init__(self,
                 path_embedding: str,
                 embedding_dim: int = 200,
                 oov_vector: str = "random", # how should the out of vocabulary(oov) word should be handled
                 seed: int = 100,
                 word_pooling: str = None
                 ):
        super().__init__()

        self.emb_dim = embedding_dim
        self.encoder = KeyedVectors.load_word2vec_format(fname = path_embedding, binary=True)
        # if we want to save the binary file into a text file
        # self.encoder.save_word2vec_format(path_embedding + "\edgarW2V.txt", binary=False)

        # handling the oov word
        if oov_vector == "random":
            np.random.seed(seed)
            self.encoder["oov"] = np.random.rand(1, embedding_dim)
        else:
            raise "not implemented"

    def forward(self, batch: Dict) -> Dict:

        max_word_batch = max(batch["n_words"])
        batch_word_embedding_list = []
        # getting embeddings of each word from stored edgar w2v embeddings
        for word_list in batch["words"]:
            len_word_list = len(word_list)
            word_embedding_list = []
            for word in word_list:
                word_value = word.value.lower()
                # if the word is present in edgarW2v model then take embeddings else embedding of oov
                if word_value in self.encoder:
                    emb = torch.from_numpy(np.copy(self.encoder[word_value]))
                    word_embedding_list.append(emb.reshape((1,-1)))
                else:
                    emb = torch.from_numpy(np.copy(self.encoder["oov"]))
                    word_embedding_list.append(emb.reshape((1,-1)))
            # creating a tensor of zeros to make all word embedding vectors of equal batch length
            num_rows = max_word_batch - len_word_list
            if num_rows == 0:
                pass
            else:
                word_embedding_list.append(torch.zeros(num_rows, self.emb_dim))

            # combining the word embedding and appending them into batch embedding list
            batch_word_embedding_list.append(torch.cat(word_embedding_list, dim=0))
        # storing embeddings in batch dict
        batch["word_embeddings"] = torch.stack(batch_word_embedding_list, dim=0).to(get_device())
        batch["token_embeddings"] = None
        batch["cls_embedding"] = None
        return batch