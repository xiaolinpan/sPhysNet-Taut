from typing import Any
import time

import torch
import torch.nn as nn

from taut_src.Networks.DimeLayers.OutputLayer import OutputLayer
from taut_src.Networks.SharedLayers.ActivationFns import activation_getter
from taut_src.utils.time_meta import record_data
from taut_src.utils.utils_functions import get_n_params


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings: Any, embedding_dim: Any) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.732, 1.732)

    def forward(self, Z: Any) -> Any:
        """

        :param Z:
        :return: m_ji: diff of bonding edge, propagated in DimeNet modules
                 v_i:  diff of atoms, propagated in PhysNet modules
                 out:  prediction of diff layer, which is part of non-bonding prediction
        """
        v_i = self.embedding(Z)

        return v_i


if __name__ == '__main__':
    model = EmbeddingLayer(95, 160, 6, 2, 3, 1)
    # for name, param in model.named_parameters():
        # print(name)
    print(get_n_params(model))
