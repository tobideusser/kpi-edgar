import logging
from typing import Dict

import numpy as np
import torch

from edgar.models.encoders import Encoder
from edgar.trainer.utils import get_device

logger = logging.getLogger(__name__)


class GloveEncoder(Encoder):
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
        # slow
        # self.encoder = KeyedVectors.load_word2vec_format(fname=path_embedding, binary=False, no_header=True)
        # fast
        self.encoder = GloveEncoder.load_glove_model(path_embedding)

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
                    word_embedding_list.append(emb.reshape((1, -1)))
                else:
                    emb = torch.from_numpy(np.copy(self.encoder["oov"]))
                    word_embedding_list.append(emb.reshape((1, -1)))
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

    @staticmethod
    def load_glove_model(file_path):
        """
        Function to load glove pre-trained embeddings
        :param file_path:
        :return:
        """

        print("Loading Glove Model")
        glove_model = {}
        with open(file_path, "r") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        return glove_model


# https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def load_glove_model(file):
    print("Loading Glove Model")
    glove_model = {}
    with open(file, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


if __name__ == "__main__":
    # this is slow because it is converting everything in its vector format
    # check_glove = KeyedVectors.load_word2vec_format(
    #     fname="/shared_with_lars_and_thiago/edgar/glove-6B/glove-6B-50d.txt", binary=False, no_header=True)
    # # this is fast but don't provide functionality similar to gensim
    # check_glove = load_glove_model("/shared_with_lars_and_thiago/edgar/glove-6B/glove-6B-50d.txt")
    #
    check = GloveEncoder(path_embedding="/shared_with_lars_and_thiago/edgar/glove-6B/glove-6B-200d.txt")
    print("check")
