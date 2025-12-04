from typing import Any
import torch
import torch.nn as nn

from taut_src.utils.utils_functions import cal_coulomb_E, floating_type


class CoulombLayer(nn.Module):
    """
    This layer is used to calculate atom-wise coulomb interaction
    """
    def __init__(self, cutoff: Any) -> None:
        super().__init__()
        self.cutoff = torch.as_tensor(cutoff).type(floating_type)

    def forward(self, qi: Any, edge_dist: Any, edge_index: Any, q_ref: Any=None, N: Any=None, atom_mol_batch: Any=None) -> Any:
        return cal_coulomb_E(qi, edge_dist, edge_index, self.cutoff, q_ref=q_ref, N=N, atom_mol_batch=atom_mol_batch)
