from typing import Any
import torch
from torch import nn

from taut_src.Networks.PhysDimeNet import PhysDimeNet
from torch.optim.swa_utils import AveragedModel
from collections import OrderedDict


def get_device() -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def fix_model_keys(state_dict: Any) -> Any:
    tmp = OrderedDict()
    for key in state_dict:
        if key.startswith("module."):
            # for some reason some module was saved with "module.module_list.*"
            tmp[key.split("module.")[-1]] = state_dict[key]
        elif key.startswith("module"):
            num = key.split(".")[0].split("module")[-1]
            tmp["main_module_list.{}.{}".format(num, ".".join(key.split(".")[1:]))] = state_dict[key]
        else:
            tmp[key] = state_dict[key]
    return tmp


class SiameseNetwork(nn.Module):
    def __init__(self, base_network: Any) -> None:
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network
    
    def forward(self, input1: Any, input2: Any) -> Any:
        output1 = self.base_network(input1)["mol_prop"][:, 1:]
        output2 = self.base_network(input2)["mol_prop"][:, 1:]
        
        diff = (output2 - output1) * 23.061
        return diff

def load_model(model_path: Any) -> Any:
    floating_type = torch.double

    net = PhysDimeNet( n_atom_embedding=95,
                         modules="P-noOut P-noOut P",
                         bonding_type="BN BN BN",
                         n_feature=160,
                         n_output=2,
                         n_dime_before_residual=1,
                         n_dime_after_residual=2,
                         n_output_dense=3,
                         n_phys_atomic_res=1,
                         n_phys_interaction_res=1,
                         n_phys_output_res=1,
                         n_bi_linear=8,
                         nh_lambda=0.01,
                         normalize=True,
                         shared_normalize_param=True,
                         activations="ssp ssp ssp",
                         restrain_non_bond_pred=True,
                         expansion_fn="(P_BN,P-noOut_BN):gaussian_64_10.0",
                         uncertainty_modify="none",
                         coulomb_charge_correct=False,
                         loss_metric="mae",
                         uni_task_ss=False,
                         lin_last=False,
                         last_lin_bias=False,
                         train_shift=True,
                         mask_z=False,
                         time_debug=False,
                         z_loss_weight=0,
                         acsf=False,
                         energy_shift=1.0,
                         energy_scale=2.0,
                         debug_mode=False,
                         action="names",
                         target_names=["gasEnergy", "waterEnergy"],
                         batch_norm=False,
                         dropout=False,
                         requires_atom_prop=False,
                         requires_atom_embedding=False,
                         pooling="sum",
                         ext_atom_features=None,
                         ext_atom_dim=0)

    siames_net = SiameseNetwork(net)
    siames_net = siames_net.to(get_device())
    siames_net = siames_net.type(floating_type)
    
    siames_net = AveragedModel( siames_net )
    state_dict = torch.load(model_path, map_location=get_device())
    # state_dict = fix_model_keys( state_dict )
    siames_net.load_state_dict(state_dict)
    siames_net = siames_net.eval()
    return siames_net


def load_models(model_paths: Any) -> Any:
    models = [load_model(model_path) for model_path in model_paths]
    return models

