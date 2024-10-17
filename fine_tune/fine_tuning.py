#!/usr/bin/env python
# coding: utf-8


import os, sys
module_path = os.path.abspath(os.path.join('..'))  # 获取上一级目录
sys.path.append(module_path)

from torch.optim.swa_utils import AveragedModel
from collections import OrderedDict

from taut_src.Networks.PhysDimeNet import PhysDimeNet
from torch.nn import Module
from torch.optim.swa_utils import AveragedModel

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from scipy.spatial import Voronoi
from torch_geometric.data import Data
from tqdm import tqdm
import os.path as osp
import os
import pandas as pd
from openbabel import pybel

import torch.nn.functional as F
from torch import nn

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem


_force_cpu = False

def fix_model_keys(state_dict):
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


def get_coords(pmol):
    coords = []
    for atom in pmol.atoms:
        coords.append(atom.coords)
    return np.array(coords)

def get_elements(pmol):
    z = []
    for atom in pmol.atoms:
        z.append(atom.atomicnum)
    return np.array(z)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

def set_force_cpu():
    """
    ONLY use it when pre-processing data
    :return:
    """
    global _force_cpu
    _force_cpu = True


def _get_index_from_matrix(num, previous_num):
    """
    get the fully-connect graph edge index compatible with torch_geometric message passing module
    eg: when num = 3, will return:
    [[0, 0, 0, 1, 1, 1, 2, 2, 2]
    [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    :param num:
    :param previous_num: the result will be added previous_num to fit the batch
    :return:
    """
    index = torch.LongTensor(2, num * num)
    index[0, :] = torch.cat([torch.zeros(num).long().fill_(i) for i in range(num)], dim=0)
    index[1, :] = torch.cat([torch.arange(num).long() for __ in range(num)], dim=0)
    mask = (index[0, :] != index[1, :])
    return index[:, mask] + previous_num


def cal_edge(R, N, prev_N, edge_index, cal_coulomb=True, short_range=True):
    """
    calculate edge distance from edge_index;
    if cal_coulomb is True, additional edge will be calculated without any restriction
    :param short_range:
    :param cal_coulomb:
    :param prev_N:
    :param edge_index:
    :param R:
    :param N:
    :return:
    """
    if cal_coulomb:
        '''
        IMPORTANT: DO NOT use num(tensor) itself as input, which will be regarded as dictionary key in this function,
        use int value(num.item())
        Using tensor as dictionary key will cause unexpected problem, for example, memory leak
        '''
        coulomb_index = torch.cat(
            [_get_index_from_matrix(num.item(), previous_num) for num, previous_num in zip(N, prev_N)], dim=-1)
        points1 = R[coulomb_index[0, :], :]
        points2 = R[coulomb_index[1, :], :]
        coulomb_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        coulomb_dist = torch.sqrt(coulomb_dist)

    else:
        coulomb_dist = None
        coulomb_index = None

    if short_range:
        short_range_index = edge_index
        points1 = R[edge_index[0, :], :]
        points2 = R[edge_index[1, :], :]
        short_range_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        short_range_dist = torch.sqrt(short_range_dist)
    else:
        short_range_dist, short_range_index = None, None
    return coulomb_dist, coulomb_index, short_range_dist, short_range_index


def scale_R(R):
    abs_min = torch.abs(R).min()
    while abs_min < 1e-3:
        R = R - 1
        abs_min = torch.abs(R).min()
    return R


def cal_msg_edge_index(edge_index):
    msg_id_1 = torch.arange(edge_index.shape[-1]).repeat(edge_index.shape[-1], 1)
    msg_id_0 = msg_id_1.t()
    source_atom = edge_index[0, :].repeat(edge_index.shape[-1], 1)
    target_atom = edge_index[1, :].view(-1, 1)
    msg_map = (source_atom == target_atom)
    result = torch.cat([msg_id_0[msg_map].view(1, -1), msg_id_1[msg_map].view(1, -1)], dim=0)
    return result


def voronoi_edge_index(R, boundary_factor, use_center):
    """
    Calculate Voronoi Diagram
    :param R: shape[-1, 3], the location of input points
    :param boundary_factor: Manually setup a boundary for those points to avoid potential error, value of [1.1, inf]
    :param use_center: If true, the boundary will be centered on center of points; otherwise, boundary will be centered
    on [0., 0., 0.]
    :return: calculated edge idx_name
    """
    R = scale_R(R)

    R_center = R.mean(dim=0) if use_center else torch.DoubleTensor([0, 0, 0])

    # maximum relative coordinate
    max_coordinate = torch.abs(R - R_center).max()
    boundary = max_coordinate * boundary_factor
    appended_R = torch.zeros(8, 3).double().fill_(boundary)
    idx = 0
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            for z_sign in [-1, 1]:
                appended_R[idx] *= torch.DoubleTensor([x_sign, y_sign, z_sign])
                idx += 1
    num_atoms = R.shape[0]

    appended_R = appended_R + R_center
    diagram = Voronoi(torch.cat([R, appended_R], dim=0), qhull_options="Qbb Qc Qz")
    edge_one_way = diagram.ridge_points
    edge_index_all = torch.LongTensor(np.concatenate([edge_one_way, edge_one_way[:, [1, 0]]], axis=0)).t()
    mask0 = edge_index_all[0, :] < num_atoms
    mask1 = edge_index_all[1, :] < num_atoms
    mask = mask0 & mask1
    edge_index = edge_index_all[:, mask]

    return edge_index


def sort_edge(edge_index):
    """
    sort the target of edge to be sequential, which may increase computational efficiency later on when training
    :param edge_index:
    :return:
    """
    arg_sort = torch.argsort(edge_index[1, :])
    return edge_index[:, arg_sort]


def mol_to_edge_index(mol):
    """
    Calculate edge_index(bonding edge) from rdkit.mol
    :param mol:
    :return:
    """
    bonds = mol.GetBonds()
    num_bonds = len(bonds)
    _edge_index = torch.zeros(2, num_bonds).long()
    for bond_id, bond in enumerate(bonds):
        _edge_index[0, bond_id] = bond.GetBeginAtomIdx()
        _edge_index[1, bond_id] = bond.GetEndAtomIdx()
    _edge_index_inv = _edge_index[[1, 0], :]
    _edge_index = torch.cat([_edge_index, _edge_index_inv], dim=-1)
    return _edge_index


def remove_bonding_edge(all_edge_index, bond_edge_index):
    """
    Remove bonding idx_name from atom_edge_index to avoid double counting
    :param all_edge_index:
    :param bond_edge_index:
    :return:
    """
    mask = torch.zeros(all_edge_index.shape[-1]).bool().fill_(False).type(all_edge_index.type())
    len_bonding = bond_edge_index.shape[-1]
    for i in range(len_bonding):
        same_atom = (all_edge_index == bond_edge_index[:, i].view(-1, 1))
        mask += (same_atom[0] & same_atom[1])
    remain_mask = ~ mask
    return all_edge_index[:, remain_mask]


def extend_bond(edge_index):
    """
    extend bond edge to a next degree, i.e. consider all 1,3 interaction as bond
    :param edge_index:
    :return:
    """
    n_edge = edge_index.size(-1)
    source = edge_index[0]
    target = edge_index[1]

    # expand into a n*n matrix
    source_expand = source.repeat(n_edge, 1)
    target_t = target.view(-1, 1)

    mask = (source_expand == target_t)
    target_index_mapper = edge_index[1].repeat(n_edge, 1)
    source_index_mapper = edge_index[0].repeat(n_edge, 1).t()

    source_index = source_index_mapper[mask]
    target_index = target_index_mapper[mask]

    extended_bond = torch.cat([source_index.view(1, -1), target_index.view(1, -1)], dim=0)
    # remove self to self interaction
    extended_bond = extended_bond[:, source_index != target_index]
    extended_bond = remove_bonding_edge(extended_bond, edge_index)
    result = torch.cat([edge_index, extended_bond], dim=-1)

    result = torch.unique(result, dim=1)
    return result


def name_extender(name, cal_3body_term=None, edge_version=None, cutoff=None, boundary_factor=None, use_center=None,
                  bond_atom_sep=None, record_long_range=False, type_3_body='B', extended_bond=False, no_ext=False,
                  geometry='QM'):
    if extended_bond:
        type_3_body = type_3_body + 'Ext'
    name += '-' + type_3_body
    if cal_3body_term:
        name += 'msg'

    if edge_version == 'cutoff':
        if cutoff is None:
            print('cutoff canot be None when edge version == cutoff, exiting...')
            exit(-1)
        name += '-cutoff-{:.2f}'.format(cutoff)
    elif edge_version == 'voronoi':
        name += '-box-{:.2f}'.format(boundary_factor)
        if use_center:
            name += '-centered'
    else:
        raise ValueError('Cannot recognize edge version(neither cutoff or voronoi), got {}'.format(edge_version))

    if sort_edge:
        name += '-sorted'

    if bond_atom_sep:
        name += '-defined_edge'

    if record_long_range:
        name += '-lr'

    name += '-{}'.format(geometry)

    if not no_ext:
        name += '.pt'
    return name


sol_keys = ["gasEnergy", "watEnergy", "octEnergy", "CalcSol", "CalcOct", "calcLogP"]


def my_pre_transform(data, edge_version, do_sort_edge, cal_efg, cutoff, boundary_factor, use_center, mol,
                     cal_3body_term, bond_atom_sep, record_long_range, type_3_body='B', extended_bond=False):
    """
    edge calculation
    atom_edge_index is non-bonding edge idx_name when bond_atom_sep=True; Otherwise, it is bonding and non-bonding together
    """
    edge_index = torch.zeros(2, 0).long()
    dist, full_edge, _, _ = cal_edge(data.pos, [data.N], [0], edge_index, cal_coulomb=True, short_range=False)
    dist = dist.cpu()
    full_edge = full_edge.cpu()

    if edge_version == 'cutoff':
        data.BN_edge_index = full_edge[:, (dist < cutoff).view(-1)]
    else:
        data.BN_edge_index = voronoi_edge_index(data.pos, boundary_factor, use_center=use_center)

    if record_long_range:
        data.L_edge_index = remove_bonding_edge(full_edge, data.BN_edge_index)

    '''
    sort edge idx_name
    '''
    if do_sort_edge:
        data.BN_edge_index = sort_edge(data.BN_edge_index)

    '''
    EFGs edge calculation
    '''
    if cal_efg:
        if edge_version == 'cutoff':
            dist, full_edge, _, _ = cal_edge(data.EFG_R, [data.EFG_N], [0], edge_index, cal_coulomb=True)
            data.EFG_edge_index = full_edge[:, (dist < cutoff).view(-1)].cpu()
        else:
            data.EFG_edge_index = voronoi_edge_index(data.EFG_R, boundary_factor, use_center=use_center)

        data.num_efg_edges = torch.LongTensor([data.EFG_edge_index.shape[-1]]).view(-1)

    if bond_atom_sep:
        '''
        Calculate bonding edges and remove those non-bonding edges which overlap with bonding edge
        '''
        if mol is None:
            print('rdkit mol file not given for molecule: {}, cannot calculate bonding edge, skipping this'.format(
                data.Z))
            return None
        B_edge_index = mol_to_edge_index(mol)
        if B_edge_index.numel() > 0 and B_edge_index.max() + 1 > data.N:
            raise ValueError('problematic mol file: {}'.format(mol))
        if B_edge_index.numel() > 0 and extended_bond:
            B_edge_index = extend_bond(B_edge_index)
        if B_edge_index.numel() > 0 and do_sort_edge:
            B_edge_index = sort_edge(B_edge_index)
        data.B_edge_index = B_edge_index
        try:
            data.N_edge_index = remove_bonding_edge(data.BN_edge_index, B_edge_index)
        except Exception as e:
            print("*"*40)
            print("BN: ", data.BN_edge_index)
            print("B: ", data.B_edge_index)
            from rdkit.Chem import MolToSmiles
            print("SMILES: ", MolToSmiles(mol))
            raise e
        _edge_list = []
        for bond_type in type_3_body:
            _edge_list.append(getattr(data, bond_type + "_edge_index"))
        _edge_index = torch.cat(_edge_list, dim=-1)
    else:
        _edge_index = data.BN_edge_index

    '''
    Calculate 3-atom term(Angle info)
    It ls essentially an "edge" of edge
    '''
    if cal_3body_term:

        atom_msg_edge_index = cal_msg_edge_index(_edge_index)
        if do_sort_edge:
            atom_msg_edge_index = sort_edge(atom_msg_edge_index)

        setattr(data, type_3_body + '_msg_edge_index', atom_msg_edge_index)

        setattr(data, 'num_' + type_3_body + '_msg_edge', torch.zeros(1).long() + atom_msg_edge_index.shape[-1])

    for bond_type in ['B', 'N', 'L', 'BN']:
        _edge_index = getattr(data, bond_type + '_edge_index', False)
        if _edge_index is not False:
            setattr(data, 'num_' + bond_type + '_edge', torch.zeros(1).long() + _edge_index.shape[-1])

    return data


def extract_mol_by_confId(mol, confId):
    mol_block = Chem.MolToMolBlock(mol, confId=confId)
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    return mol


def generate_confs(smi, numConfs=1):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    
    ps = AllChem.ETKDG()
    ps.maxAttempts = 1000
    ps.randomSeed = 1
    ps.pruneRmsThresh = 0.1
    ps.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs, ps)
    
    confs = []
    for cid in cids:
        mol_conf = extract_mol_by_confId(mol, cid)
        confs.append(mol_conf)
    return confs


def optimize(mol):
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0)
    ff.Initialize()
    ff.Minimize(maxIts=1000)
    E = ff.CalcEnergy()
    return E


def get_low_energy_conf(smi, num_confs):
    mol_confs = generate_confs(smi, num_confs)
    data = []
    for m in mol_confs:
        E = optimize(m)
        data.append([E, m])
    sdata = sorted(data, key=lambda x: x[0])
    low_energy, opt_conf = sdata[0]
    mol_block = Chem.MolToMolBlock(opt_conf)
    pmol = pybel.readstring("mol", mol_block)
    return pmol


def calc_data( smi ):
    pmol = get_low_energy_conf(smi, num_confs=300)
   
    coords = get_coords(pmol)
    elements = get_elements(pmol)
   
    N = coords.shape[0]

    this_data = Data(pos = torch.as_tensor(coords, dtype=torch.double),
                     Z = torch.as_tensor(elements, dtype=torch.long),
                     N = torch.as_tensor(N, dtype=torch.long).view(-1),
                     BN_edge_index_correct = torch.tensor([0], dtype=torch.long))
    
    nthis_data = my_pre_transform( this_data, edge_version="cutoff", 
                                   do_sort_edge=True, cal_efg=False,
                                   cutoff=10.0, boundary_factor=100., use_center=True, 
                                   mol=None, cal_3body_term=False,
                                   bond_atom_sep=False, record_long_range=True)
    return nthis_data
    
   
def get_dft_pmol(name, index):
    fname = name + "_t" + str(index)
    dff = df_gas[df_gas["name"] == fname]
    E_gas = dff.iloc[0, 1]
    return E_gas * 627.51


class EmaAmsGrad(torch.optim.Adam):
    def __init__(self, training_model: torch.nn.Module, lr=1e-3, betas=(0.9, 0.99),
                 eps=1e-8, weight_decay=0, ema=0.999, shadow_dict=None):
        super().__init__(filter(lambda p: p.requires_grad, training_model.parameters()), lr, betas, eps, weight_decay, amsgrad=True)
        # for initialization of shadow model
        self.shadow_dict = shadow_dict
        self.ema = ema
        self.training_model = training_model

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            return ema * averaged_model_parameter + (1 - ema) * model_parameter

        def avg_fn_deactivated(averaged_model_parameter, model_parameter, num_averaged):
            return model_parameter

        self.deactivated = (ema < 0)
        self.shadow_model = AveragedModel(training_model, device=get_device(),
                                          avg_fn=avg_fn_deactivated if self.deactivated else avg_fn)

    def step(self, closure=None):
        # t0 = time.time()

        loss = super().step(closure)

        # t0 = record_data('AMS grad', t0)
        if self.shadow_model.n_averaged == 0 and self.shadow_dict is not None:
            self.shadow_model.module.load_state_dict(self.shadow_dict, strict=False)
            self.shadow_model.n_averaged += 1
        else:
            self.shadow_model.update_parameters(self.training_model)

        # t0 = record_data('shadow update', t0)
        return loss


class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network
    
    def forward(self, input1, input2):
        output1 = self.base_network(input1)["mol_prop"]
        output2 = self.base_network(input2)["mol_prop"]
        
        diff = (output2 - output1) * 23.061
        return diff


def load_model(model_path):
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
                         requires_atom_embedding=True,
                         pooling="sum",
                         ext_atom_features=None,
                         ext_atom_dim=0)
    
    state_dict = torch.load(model_path, map_location=get_device())
    state_dict = fix_model_keys(state_dict)
    net.load_state_dict(state_dict=state_dict, strict=False)

    siamese_net = SiameseNetwork(net)
    siamese_net = siamese_net.type(floating_type)
    siamese_net = siamese_net.to(get_device())
    return siamese_net


def loss_fn(output, ddG):
    loss = F.l1_loss(output, ddG)
    return loss


@torch.no_grad()
def valid_fn(data_loader, model):
    model.eval()

    trues, preds = [], []
    for data1, data2, ddG_g, ddG_w in data_loader:
        data1 = data1.to(get_device())
        data2 = data2.to(get_device())
        ddG_w = ddG_w.to(get_device()).view(-1, 1).double()
        
        output = model(data1, data2)[:, 1:]
        preds.extend(output.view(-1).cpu().numpy().tolist())
        trues.extend(ddG_w.view(-1).cpu().numpy().tolist())

    all_mae = mean_absolute_error(trues, preds)
    all_rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    Rp= pearsonr(trues, preds)[0]
    return Rp, r2, all_mae, all_rmse


def train_step( data1, 
                data2, 
                ddG_gas, 
                ddG_water, 
                model,
                optimizer):
    optimizer.zero_grad()
    
    data1 = data1.to(get_device())
    data2 = data2.to(get_device())
    ddG_water = ddG_water.to(get_device()).view(-1, 1).double()
    ddG_gas = ddG_gas.to(get_device()).view(-1, 1).double()
    output = model(data1, data2)
    
    loss = 0.8*loss_fn(output[:, 1:], ddG_water) + 0.2*loss_fn(output[:, :1], ddG_gas)
    loss.backward()
    optimizer.step()
    return loss


def init_model(model_path):
    siamese_net = load_model( model_path )
    
    for param in siamese_net.parameters():
        param.requires_grad = False
    
    for param in siamese_net.base_network.main_module_list[2].parameters():
        param.requires_grad = True
    return siamese_net


def split_dataset(dataset):
    set_0, set_1, set_2, set_3 = [], [], [], []
    for pair in dataset:
        set_0.append( pair[0] )
        set_1.append( pair[1] )
        set_2.append( pair[2] )
        set_3.append( pair[3] )
    return set_0, set_1, set_2, set_3


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, datasetA, datasetB, datasetC, datasetD):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC
        self.datasetD = datasetD
        

    def __getitem__(self, idx):
        return self.datasetA[idx], self.datasetB[idx], self.datasetC[idx], self.datasetD[idx]

    def __len__(self):
            return len(self.datasetA)


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    return batchA, batchB, batchC, batchD


def construct_pair_dataloader(dataset, batch_size, shuffle):
    set_0, set_1, set_2, set_3 = split_dataset( dataset )
    pair_dataset = PairDataset( set_0, set_1, set_2, set_3 )

    loader = DataLoader(pair_dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=0,  
                        collate_fn=collate)
    return loader


def prepare_dataloader( df,  batch_size, shuffle ):
    datasets = []
    for idx, name, smi1, smi2, ddG_w in df.itertuples():
       nthis_data1 = calc_data(smi1)
       nthis_data2 = calc_data(smi2)
       E_gas2 = get_dft_pmol(name, index=2)
       E_gas1 = get_dft_pmol(name, index=1)
       ddG_g = E_gas2-E_gas1
       datasets.append( [nthis_data1, nthis_data2, ddG_g, ddG_w] )
    print("Dataset generating done, it contains {} samples.".format( len(datasets) ))
    
    dataloader = construct_pair_dataloader(datasets, batch_size, shuffle)
    return dataloader


if __name__=="__main__":
    pretrain_model_path = "models/best_model.pt"
    siamese_net = init_model(pretrain_model_path)
    
    optimizer = EmaAmsGrad( siamese_net, 
                            weight_decay=0.001, 
                            lr=0.0001, ema=0.999, 
                            shadow_dict=siamese_net.state_dict())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=100)
    
    
    folder = sys.argv[1]
    df_gas = pd.read_csv("tautobase_gas_phase.csv")
    df_train = pd.read_csv(os.path.join(folder, "fine_tune_train.csv")).loc[:, ['name', 'smiles0', 'smiles1', 'ddG']]
    df_test = pd.read_csv(os.path.join( folder, "fine_tune_valid.csv")).loc[:, ['name', 'smiles0', 'smiles1', 'ddG']]
    
    train_dataloader = prepare_dataloader( df_train, batch_size=64, shuffle=False )
    valid_dataloader = prepare_dataloader( df_test, batch_size=16, shuffle=False )
    
    
    run_directory = os.path.join( folder, "models" )
    
    best_loss = np.inf
    for epoch in range(2000):
        siamese_net.train()
        
        train_loss = 0.
        for data1, data2, ddG_gas, ddG_water in train_dataloader:
            loss = train_step(data1, data2, ddG_gas, ddG_water, siamese_net, optimizer)
            train_loss += loss.item() * data1.num_graphs
    
            loss = train_step(data2, data1, -1.0 * ddG_gas, -1.0 * ddG_water, siamese_net, optimizer)
            train_loss += loss.item() * data1.num_graphs
            
        shadow_net = optimizer.shadow_model
        
        val_Rp, val_r2, val_mae, val_rmse = valid_fn(valid_dataloader, shadow_net)
        print( "Testing Epoch: {}, Rp: {:.3f}, R2: {:.3f}, MAE: {:.3f}, RMSE: {:.3f}\n".format( epoch, 
                                                                                                val_Rp, 
                                                                                                val_r2, 
                                                                                                val_mae, 
                                                                                                val_rmse))
    
        scheduler.step(val_rmse)
        
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(shadow_net.state_dict(), osp.join(run_directory, 'best_model.pt'))
            torch.save(shadow_net.state_dict(), osp.join(run_directory, 'training_model.pt'))
            torch.save(optimizer.state_dict(), osp.join(run_directory, 'best_model_optimizer.pt'))
            torch.save(scheduler.state_dict(), osp.join(run_directory, "best_model_scheduler.pt"))


