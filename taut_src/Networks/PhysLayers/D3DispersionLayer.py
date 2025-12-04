from typing import Any
import warnings
import torch.nn as nn

from taut_src.utils.grimme_d3 import edisp, d3_autoang


class D3DispersionLayer(nn.Module):
    def __init__(self, s6: Any, s8: Any, a1: Any, a2: Any) -> None:
        warnings.warn("D3 dispersion algorithm is unstable when molecule grows larger (0./0. when calculating c6 coe.)",
                      DeprecationWarning)
        super().__init__()
        self.a2 = a2
        self.a1 = a1
        self.s8 = s8
        self.s6 = s6

    def forward(self, Z: Any, edge_dist: Any, edge_index: Any) -> Any:
        E_atom_d3 = edisp(Z, edge_dist/d3_autoang, idx_i=edge_index[0, :], idx_j=edge_index[1, :],
                          s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2)
        return E_atom_d3.view(-1)
