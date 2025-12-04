from typing import Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional
import torch_geometric
from torch import Tensor
from torch_sparse import SparseTensor

from taut_src.utils.utils_functions import semi_orthogonal_glorot_weights, floating_type
from taut_src.Networks.SharedLayers.ResidualLayer import ResidualLayer
from taut_src.Networks.SharedLayers.ActivationFns import activation_getter


class InteractionModule(nn.Module):
    """
    The interaction layer defined in PhysNet
    """

    def __init__(self, F: Any, K: Any, n_res_interaction: Any, activation: Any, batch_norm: Any, dropout: Any) -> None:
        super().__init__()
        u = torch.Tensor(1, F).type(floating_type).fill_(1.)
        self.register_parameter('u', torch.nn.Parameter(u, True))

        self.message_pass_layer = MessagePassingLayer(aggr='add', F=F, K=K, activation=activation,
                                                      batch_norm=batch_norm, dropout=dropout)

        self.n_res_interaction = n_res_interaction
        for i in range(n_res_interaction):
            self.add_module('res_layer' + str(i), ResidualLayer(F=F, activation=activation, batch_norm=batch_norm,
                                                                dropout=dropout))

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(F, momentum=1.)
        self.lin_last = nn.Linear(F, F)
        self.lin_last.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_last.bias.data.zero_()

        self.activation = activation_getter(activation)

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor], edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        msged_x = self.message_pass_layer(x, edge_index, edge_attr)
        tmp_res = msged_x
        for i in range(self.n_res_interaction):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res)
        if self.batch_norm:
            tmp_res = self.bn(tmp_res)
        v = self.activation(tmp_res)
        v = self.lin_last(v)
        return v + torch.mul(x, self.u), msged_x


class MessagePassingLayer(torch_geometric.nn.MessagePassing):
    """
    message passing layer in torch_geometric
    see: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html for more details
    """

    def __init__(self, F: Any, K: Any, activation: Any, aggr: Any, batch_norm: Any, dropout: Any) -> None:
        self.batch_norm = batch_norm
        flow = 'source_to_target'
        super().__init__(aggr=aggr, flow=flow)
        self.lin_for_same = nn.Linear(F, F)
        self.lin_for_same.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_for_same.bias.data.zero_()

        self.lin_for_diff = nn.Linear(F, F)
        self.lin_for_diff.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_for_diff.bias.data.zero_()

        if self.batch_norm:
            self.bn_same = nn.BatchNorm1d(F, momentum=1.)
            self.bn_diff = nn.BatchNorm1d(F, momentum=1.)

        self.G = nn.Linear(K, F, bias=False)
        self.G.weight.data.zero_()

        self.activation = activation_getter(activation)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.batch_norm:
            x_j = self.bn_diff(x_j)
        msg = self.lin_for_diff(x_j)
        msg = self.activation(msg)
        masked_edge_attr = self.G(edge_attr)
        msg = torch.mul(msg, masked_edge_attr)
        return msg

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor], edge_attr: Tensor) -> Tensor:
        x = self.activation(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        if self.batch_norm:
            x = self.bn_same(x)
        a = self.activation(self.lin_for_same(x))
        return a + aggr_out


if __name__ == '__main__':
    pass
