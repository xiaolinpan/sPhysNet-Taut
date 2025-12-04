from typing import Any, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor

from taut_src.Networks.SharedLayers.ActivationFns import activation_getter
from taut_src.utils.utils_functions import floating_type, get_n_params


class DimeNetMPN(MessagePassing):

    def __init__(self, n_tensor: Any, dim_msg: Any, dim_rbf: Any, dim_sbf: Any, activation: Any) -> None:
        super().__init__()
        self.n_tensor = n_tensor
        self.dim_msg = dim_msg
        self.lin_source = nn.Linear(dim_msg, dim_msg)
        self.lin_target = nn.Linear(dim_msg, dim_msg)
        self.lin_rbf = nn.Linear(dim_rbf, dim_msg, bias=False)
        self.lin_sbf = nn.Linear(dim_sbf, n_tensor, bias=False)

        '''
        registering bi-linear layer weight (without bias)
        '''
        W_bi_linear = torch.zeros(dim_msg, dim_msg, n_tensor).type(floating_type).uniform_(-1/dim_msg, 1/dim_msg)
        self.register_parameter('W_bi_linear', torch.nn.Parameter(W_bi_linear, requires_grad=True))

        self.activation = activation_getter(activation)

    def message(self, x_j: Tensor, rbf_j: Tensor, edge_attr: Tensor) -> Tensor:
        # t0 = time.time()

        x_j = self.activation(self.lin_source(x_j))

        # t0 = record_data('main.message-passing.lin-source', t0)

        rbf = self.lin_rbf(rbf_j)

        # t0 = record_data('main.message-passing.lin-rbf', t0)

        msg1 = x_j * rbf

        # t0 = record_data('main.message-passing.msg1', t0)

        sbf = self.lin_sbf(edge_attr)

        # t0 = record_data('main.message-passing.lin-sbf', t0)

        msg = torch.einsum('wi,wl,ijl->wj', msg1, sbf, self.W_bi_linear)

        # t0 = record_data('main.message-passing.bi-linear', t0)
        return msg

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        x = self.activation(self.lin_target(x))
        return x + aggr_out

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor], rbf: Tensor, sbf: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, rbf=rbf, edge_attr=sbf)


if __name__ == '__main__':
    model = DimeNetMPN(8, 128, 6, 5, 'swish')
    print(get_n_params(model))
