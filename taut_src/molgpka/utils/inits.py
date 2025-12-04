from typing import Any
import math


def uniform(size: Any, tensor: Any) -> Any:
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor: Any, fan: Any, a: Any) -> Any:
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor: Any) -> Any:
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor: Any) -> Any:
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor: Any) -> Any:
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor: Any, mean: Any, std: Any) -> Any:
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn: Any) -> Any:
    def _reset(item: Any) -> Any:
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
