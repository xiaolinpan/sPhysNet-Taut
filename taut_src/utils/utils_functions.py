from typing import Any
import copy
import gc
import logging
import math
import os
import random
import re
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch_geometric
from ase import Atoms
from scipy.spatial import KDTree
from torch.optim.swa_utils import AveragedModel
from torch_scatter import scatter

# Constants:

# Coulomb’s constant in eV A and e
from taut_src.utils.time_meta import record_data

k_e = 14.399645352
# CPU device, part of the functions are calculated faster in cpu
cpu_device = torch.device('cpu')
# Convert Hartree to eV
hartree2ev = 27.2114
# floating type
floating_type = torch.double

kcal2ev = 1 / 23.06035
# Atomic reference energy at 0K (unit: Hartree)
atom_ref = {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730}

matrix_to_index_map = {}

mae_fn = torch.nn.L1Loss(reduction='mean')
mse_fn = torch.nn.MSELoss(reduction='mean')

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device() -> Any:
    # we use a function to get device for proper distributed training behaviour
    # return torch.device("cpu")
    return _device


def find_free_port() -> Any:
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _cutoff_fn(D: Any, cutoff: Any) -> Any:
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def gaussian_rbf(D: Any, centers: Any, widths: Any, cutoff: Any, return_dict: Any=False) -> Any:
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
    if return_dict:
        return {"rbf": rbf}
    else:
        return rbf


def _get_index_from_matrix(num: Any, previous_num: Any) -> Any:
    """
    get the edge index compatible with torch_geometric message passing module
    eg: when num = 3, will return:
    [[0, 0, 0, 1, 1, 1, 2, 2, 2]
    [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    :param num:
    :param previous_num: the result will be added previous_num to fit the batch
    :return:
    """
    if num in matrix_to_index_map.keys():
        return matrix_to_index_map[num] + previous_num
    else:
        index = torch.LongTensor(2, num * num).to(get_device())
        index[0, :] = torch.cat([torch.zeros(num, device=get_device()).long().fill_(i) for i in range(num)], dim=0)
        index[1, :] = torch.cat([torch.arange(num, device=get_device()).long() for _ in range(num)], dim=0)
        mask = (index[0, :] != index[1, :])
        matrix_to_index_map[num] = index[:, mask]
        return matrix_to_index_map[num] + previous_num


matrix_modify = {}


def _get_modify_matrix(num: Any) -> Any:
    """
    get the modify matrix.
    equivalent to -torch.eye(num)
    data will be stored in matrix_modify to save time when next time need it
    :param num:
    :return:
    """
    if num in matrix_modify.keys():
        return matrix_modify[num]
    else:
        matrix = torch.Tensor(num, num).type(floating_type).zero_()
        for i in range(num):
            matrix[i, i] = -1.
        matrix_modify[num] = matrix
        return matrix


batch_pattern = {}


def _get_batch_pattern(batch_size: Any, max_num: Any) -> Any:
    """
    get the batch pattern, for example, if batch_size=5, max_num=3
    the pattern will be: [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
    new pattern will be stored in batch_pattern dictionary to avoid recalculation
    :return:
    """
    if batch_size in batch_pattern.keys():
        return batch_pattern[batch_size]
    else:
        pattern = [i // max_num for i in range(batch_size * max_num)]
        batch_pattern[batch_size] = pattern
        return pattern


def _cal_dist(d1: Any, d2: Any) -> Any:
    """
    calculate the Euclidean distance between d1 and d2
    :param d1:
    :param d2:
    :return:
    """
    delta_R = d1 - d2
    return torch.sqrt(torch.sum(torch.mul(delta_R, delta_R))).view(-1, 1).type(floating_type)


def softplus_inverse(x: Any) -> Any:
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    return torch.log(-torch.expm1(-x)) + x


def _cutoff_fn(D: Any, cutoff: Any) -> Any:
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def _chi_ij(R_ij: Any, cutoff: Any) -> Any:
    """
    Chi(Rij) function which is used to calculate long-range energy
    return 0 when R_ij = -1 (use -1 instead of 0 to prevent nan when backward)
    :return: Chi(Rij)
    """
    return torch.where(R_ij != -1, _cutoff_fn(2 * R_ij, cutoff) / torch.sqrt(torch.mul(R_ij, R_ij) + 1) +
                       (1 - _cutoff_fn(2 * R_ij, cutoff)) / R_ij, torch.zeros_like(R_ij))


def _correct_q(qi: Any, N: Any, atom_to_mol_batch: Any, q_ref: Any) -> Any:
    """
    calculate corrected partial_q in PhysNet
    :param qi: partial charge predicted by PhysNet, shape(-1, 1)
    :return: corrected partial_q, shape(-1, 1)
    """
    # predicted sum charge
    Q_pred = scatter(reduce='add', src=qi, index=atom_to_mol_batch, dim=0)

    correct_term = (Q_pred - q_ref) / (N.type(floating_type).to(get_device()))
    # broad cast according to batch to make sure dim correct
    broadcasted_correct_term = correct_term.take(atom_to_mol_batch)
    return qi - broadcasted_correct_term


def cal_coulomb_E(qi: torch.Tensor, edge_dist: Any, edge_index: Any, cutoff: Any, q_ref: Any, N: Any, atom_mol_batch: Any) -> Any:
    """
    Calculate coulomb Energy from chi(Rij) and corrected q
    Calculate ATOM-WISE energy!
    :return: calculated E
    """
    # debug: cal passed time to improve code efficiency
    # print('&' * 40)
    # t0 = time.time()

    cutoff = cutoff.to(get_device())

    # debug: cal passed time to improve code
    # print('T--------pre cal: ', time.time() - t0)
    if q_ref is not None:
        """
        This correction term is from PhysNet paper:
        As neural networks are a purely numerical algorithm, it is not guaranteed a priori
        that the sum of all predicted atomic partial charges qi is equal
        to the total charge Q (although the result is usually very close
        when the neural network is properly trained), so a correction
        scheme like eq 14 is necessary to guarantee charge
        conservation
        """
        assert N is not None
        assert atom_mol_batch is not None
        qi = _correct_q(qi, N, atom_mol_batch, q_ref)

    # Qi will be corrected according to the paper
    # qi = _partial_q(qi, N, atom_to_mol_batch, Q)

    # debug: cal passed time to improve code
    # print('T--------cal partial qi: ', time.time() - t0)

    # debug: cal passed time to improve code
    # print('T--------split: ', time.time() - t0)

    # Probably I should do qi = qi.clone() to avoid inplace calculation
    q_first = qi.take(edge_index[0, :]).view(-1, 1)
    q_second = qi.take(edge_index[1, :]).view(-1, 1)
    revised_dist = _chi_ij(edge_dist, cutoff=cutoff)
    coulomb_E_terms = q_first * revised_dist * q_second
    '''
    set dim_size here in case the last batch has only one 'atom', which will cause dim to be 1 less because no 
    edge will be formed th that way 
    '''
    coulomb_E = scatter(reduce='add', src=coulomb_E_terms.view(-1), index=edge_index[0, :], dim_size=qi.shape[0], dim=0)

    # debug: cal passed time to improve code
    # print('T--------for loop: ', time.time() - t0)
    # print('&' * 40)

    # times 1/2 because of the double counting
    return (coulomb_E / 2).to(get_device())


def cal_p(qi: Any, R: Any, atom_to_mol_batch: Any) -> Any:
    """
    Calculate pi from qi and molecule coordinate
    :return: pi
    """

    tmp = torch.mul(qi.view(-1, 1), R.to(get_device()))
    p = scatter(reduce='add', src=tmp, index=atom_to_mol_batch.to(get_device()), dim=0)
    return p


def cal_edge(R: Any, N: Any, prev_N: Any, edge_index: Any, cal_coulomb: Any=True) -> Any:
    """
    calculate edge distance from edge_index;
    if cal_coulomb is True, additional edge will be calculated without any restriction
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

    short_range_index = edge_index
    points1 = R[edge_index[0, :], :]
    points2 = R[edge_index[1, :], :]
    short_range_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
    short_range_dist = torch.sqrt(short_range_dist)
    return coulomb_dist, coulomb_index, short_range_dist, short_range_index


def get_batch(atom_map: Any, max_num: Any) -> Any:
    """
    from map to batch
    :param atom_map:
    :param atom_map, batch_size, max_num:
    :return: batch, example: [0,0,0,0,0,0,1,1,2,2,2,...]
    """
    batch_size = atom_map.shape[0] // max_num
    pattern = _get_batch_pattern(batch_size, max_num)
    return torch.LongTensor(pattern)[atom_map]


def get_uniform_variance(n1: Any, n2: Any) -> Any:
    """
    get the uniform variance to initialize the weight of DNNs suggested at
    Glorot,X.;Bengio,Y. Understanding the Difficulty of Training Deep Feed forward Neural Networks.
    Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 2010; pp 249–256.
    :param n1: the size of previous layer
    :param n2: the size of next layer :return: uniform variance
    """
    return math.sqrt(6) / math.sqrt(n1 + n2)


# generates a random square orthogonal matrix of dimension dim
def square_orthogonal_matrix(dim: Any=3, seed: Any=None) -> Any:
    random_state = np.random
    if seed is not None:  # allows to get the same matrix every time
        random_state.seed(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# generates a random (semi-)orthogonal matrix of size NxM
def semi_orthogonal_matrix(N: Any, M: Any, seed: Any=None) -> Any:
    if N > M:  # number of rows is larger than number of columns
        square_matrix = square_orthogonal_matrix(dim=N, seed=seed)
    else:  # number of columns is larger than number of rows
        square_matrix = square_orthogonal_matrix(dim=M, seed=seed)
    return square_matrix[:N, :M]


# generates a weight matrix with variance according to Glorot initialization
# based on a random (semi-)orthogonal matrix
# neural networks are expected to learn better when features are decorrelated
# (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
# "Dropout: a simple way to prevent neural networks from overfitting",
# "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
def semi_orthogonal_glorot_weights(n_in: Any, n_out: Any, scale: Any=2.0, seed: Any=None) -> Any:
    W = semi_orthogonal_matrix(n_in, n_out, seed=seed)
    W *= np.sqrt(scale / ((n_in + n_out) * W.var()))
    return torch.Tensor(W).type(floating_type).t()


def get_atom_to_efgs_batch(efgs_batch: Any, num_efgs: Any, atom_to_mol_mask: Any) -> Any:
    # efgs_batch: shape(n_batch, 29)
    # num_efgs: shape(n_batch)

    _batch_corrector = torch.zeros_like(num_efgs)
    _batch_corrector[1:] = num_efgs[:-1]
    batch_size = _batch_corrector.shape[0]
    for i in range(1, batch_size):
        _batch_corrector[i] = _batch_corrector[i - 1] + _batch_corrector[i]
    _batch_corrector = _batch_corrector.view(-1, 1)  # make sure correct dimension for broadcasting
    efgs_batch = efgs_batch + _batch_corrector
    atom_to_efgs_batch = efgs_batch.view(-1)[atom_to_mol_mask]
    return atom_to_efgs_batch


def get_kd_tree_array(R: Any, N: Any) -> Any:
    """
    Used in data_provider, encapsulate coordinates and numbers into a kd_tree array(tensor)
    :param R: Coordinates
    :param N: Number of atoms in this molecule
    :return: tensor of KD_Tree instances
    """
    num_molecules = R.shape[0]
    kd_trees = np.empty(num_molecules, dtype=KDTree)
    for i in range(num_molecules):
        kd_trees[i] = KDTree(R[i, :N[i].item(), :])
    return kd_trees


def atom_mean_std(E: Any, N: Any, index: Any) -> Any:
    """
    calculate the mean and stand variance of Energy in the training set
    :return:
    """
    mean = 0.0
    std = 0.0
    num_mol = len(index)
    for _i in range(num_mol):
        i = index[_i]
        m_prev = mean
        x = E[i] / N[i]
        mean += (x - mean) / (i + 1)
        std += (x - mean) * (x - m_prev)
    std = math.sqrt(std / num_mol)
    return mean, std


def _pre_nums(N: Any, i: Any) -> Any:
    return N[i - 1] if i > 0 else 0


def load_data_from_index(dataset: Any, indexes: Any) -> Any:
    """
    Similar to Batch.to_data_list() method in torch_geometric
    :param indexes:
    :param dataset:
    :return:
    """
    result = collate_fn([dataset[i.item()] for i in indexes], clone=False)
    return result


def _cal_dim(key: Any) -> Any:
    return -1 if re.search("index", key) else 0


def collate_fn(data_list: Any, clone: Any=False, cal_acsf: Any=False) -> Any:
    """
    Note: using clone here, maybe not efficient
    :param cal_acsf: calculate ACSF
    :param clone:
    :param data_list:
    :return:
    """
    # torch.cuda.synchronize(device)
    # t0 = time.time()

    if clone:
        data_list = [data.clone() for data in data_list]

    for data in data_list:
        if hasattr(data, "acsf"):
            data.acsf = torch.as_tensor(data.acsf)
        else:
            break

    batch = torch_geometric.data.Data()
    batch_size = len(data_list)
    keys = data_list[0].keys

    for key in keys:
        b4_cat = [data_list[i][key] for i in range(batch_size)]
        if isinstance(b4_cat[0], torch.Tensor):
            batch[key] = torch.cat(b4_cat, dim=_cal_dim(key))
        else:
            batch[key] = b4_cat

    cum = {}
    for key in keys:
        if re.search('num_', key) or (key == 'N'):
            cum_sum_before = torch.zeros_like(batch[key])
            cum_sum_before[1:] = torch.cumsum(batch[key], dim=0)[:-1]
            # key[4:] means:   num_B_edge -> B_edge
            cum_name = 'N' if key == 'N' else key[4:]
            cum[cum_name] = cum_sum_before
    for key in keys:
        if re.search('edge_index', key):
            # BN_edge_index -> BN_edge
            # B_msg_edge_index -> B_msg_edge
            edge_name = key[:-6]
            if re.search('_msg_edge_index', key):
                # B_msg_edge_index -> B
                bond_type = key[:-15]
                batch[key + '_correct'] = torch.repeat_interleave(cum[bond_type + '_edge'], batch['num_' + edge_name])
            else:
                batch[key + '_correct'] = torch.repeat_interleave(cum['N'], batch['num_' + edge_name])
    batch.atom_mol_batch = torch.repeat_interleave(torch.arange(batch['N'].shape[0]), batch['N'])

    # t0 = record_data('collate fn', t0)
    return batch.contiguous()


def dime_edge_expansion(R: Any, edge_index: Any, msg_edge_index: Any, n_dime_rbf: Any, dist_calculator: Any, bessel_calculator: Any,
                        feature_interact_dist: Any, cos_theta: Any=True, return_dict: Any=False, **kwargs: Any) -> Any:
    # t0 = record_data('edge_msg_gen.load_data', t0)

    """
    calculating bonding infos
    those data will be used in DimeNet modules
    """
    dist_atom = dist_calculator(R[edge_index[0, :], :], R[edge_index[1, :], :])
    rbf_ji = bessel_calculator.cal_rbf(dist_atom, feature_interact_dist, n_dime_rbf)

    # t0 = record_data('edge_msg_gen.bond_rbf', t0)

    dist_msg = dist_calculator(
        R[edge_index[0, msg_edge_index[1, :]], :], R[edge_index[1, msg_edge_index[1, :]], :]
    ).view(-1, 1)
    angle_msg = cal_angle(
        R, edge_index[:, msg_edge_index[0, :]], edge_index[:, msg_edge_index[1, :]], cos_theta).view(-1, 1)
    sbf_kji = bessel_calculator.cal_sbf(dist_msg, angle_msg, feature_interact_dist)

    # t0 = record_data('edge_msg_gen.bond_sbf', t0)
    if return_dict:
        return {"rbf_ji": rbf_ji, "sbf_kji": sbf_kji}
    else:
        return rbf_ji, sbf_kji


def get_n_params(model: Any, logger: Any=None, only_trainable: Any=False) -> Any:
    """
    Calculate num of parameters in the model
    :param only_trainable: Only count trainable
    :param logger:
    :param model:
    :return:
    """
    result = ''
    counted_params = []
    for name, param in model.named_parameters():
        if not (only_trainable and not param.requires_grad):
            if logger is not None:
                logger.info('{}: {}'.format(name, param.data.shape))
            result = result + '{}: {}\n'.format(name, param.data.shape)
            counted_params.append(param)
    return sum([x.nelement() for x in counted_params]), result


def cal_angle(R: Any, edge1: Any, edge2: Any, cal_cos_theta: Any) -> Any:
    delta_R1 = R[edge1[0, :], :] - R[edge1[1, :], :]
    delta_R2 = R[edge2[0, :], :] - R[edge2[1, :], :]
    inner = torch.sum(delta_R1 * delta_R2, dim=-1)
    delta_R1_l = torch.sqrt(torch.sum(delta_R1 ** 2, dim=-1))
    delta_R2_l = torch.sqrt(torch.sum(delta_R2 ** 2, dim=-1))
    cos_theta = inner / (delta_R1_l * delta_R2_l + 1e-7)
    if cal_cos_theta:
        angle = cos_theta
    else:
        angle = torch.acos(cos_theta)
    return angle.view(-1, 1)


def get_tensors() -> Any:
    """
    print out tensors in current system to debug memory leak
    :return: set of infos about tensors
    """
    result = {"set_init"}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tup = (obj.__hash__(), obj.size())
                result.add(tup)
        except:
            pass
    print('*' * 30)
    return result


def get_lr(optimizer: Any) -> Any:
    for param_group in optimizer.param_groups:
        return param_group['lr']


def info_resolver(s: Any) -> Any:
    """
    Internal function which resolve expansion function into details, eg:
    gaussian_64_10.0 means gaussian expansion, n=64 and cutoff=10.0
    :param s:
    :return:
    """
    info = s.split('_')
    result = {'name': info[0]}
    if info[0] == 'bessel' or info[0] == 'gaussian':
        result['n'] = int(info[1])
        result['dist'] = float(info[2])
    elif info[0] == 'defaultDime':
        result['n'] = int(info[1])
        result['envelop_p'] = int(info[2])
        result['n_srbf'] = int(info[3])
        result['n_shbf'] = int(info[4])
        result['dist'] = float(info[5])
    elif info[0] == 'coulomb':
        result['dist'] = float(info[1])
    elif info[0] == 'none':
        pass
    else:
        raise ValueError(f"Invalid expansion function {s} !")
    return result


def expansion_splitter(s: Any) -> Any:
    """
    Internal use only
    Strip expansion function into a dictionary
    :param s:
    :return:
    """
    result = {}
    for mapping in s.split(' '):
        value = mapping.split(':')[1]
        keys = mapping.split(':')[0]
        if keys[0] == '(':
            assert keys[-1] == ')'
            keys = keys[1:-1]
            for key in keys.split(','):
                result[key.strip()] = value
        else:
            result[keys.strip()] = value
    return result


def error_message(value: Any, name: Any) -> Any:
    raise ValueError('Invalid {} : {}'.format(name, value))


def print_val_results(dataset_name: Any, loss: Any, emae: Any, ermse: Any, qmae: Any, qrmse: Any, pmae: Any, prmse: Any) -> Any:
    log_info = 'Validating {}: '.format(dataset_name)
    log_info += (' loss: {:.6f} '.format(loss))
    log_info += ('emae: {:.6f} '.format(emae))
    log_info += ('ermse: {:.6f} '.format(ermse))
    log_info += ('qmae: {:.6f} '.format(qmae))
    log_info += ('qrmse: {:.6f} '.format(qrmse))
    log_info += ('pmae: {:.6f} '.format(pmae))
    log_info += ('prmse: {:.6f} '.format(prmse))
    return log_info


def option_solver(option_txt: Any, type_conversion: Any=False, return_base: Any=False) -> Any:
    option_base = option_txt.split('[')[0]
    if len(option_txt.split('[')) == 1:
        result = {}
    else:
        # option_txt should be like :    '[n_read_out=2,other_option=value]'
        # which will be converted into a dictionary: {n_read_out: 2, other_option: value}
        option_txt = option_txt.split('[')[1]
        option_txt = option_txt[:-1]
        result = {argument.split('=')[0].strip(): argument.split('=')[1].strip()
                  for argument in option_txt.split(',')}
        if type_conversion:
            for key in result.keys():
                value_final = copy.copy(result[key])
                try:
                    tmp = float(value_final)
                    result[key] = tmp
                except ValueError:
                    pass

                try:
                    tmp = int(value_final)
                    result[key] = tmp
                except ValueError:
                    pass

                if result[key] in ["True", "False"]:
                    result[key] = (result[key] == "True")
    if return_base:
        return option_base, result
    else:
        return result


def preprocess_config(config_dict: Any) -> Any:
    config_dict = copy.deepcopy(config_dict)
    for bool_key in ["debug_mode", "auto_sol", "reset_optimizer", "target_nodes", "reset_output_layers",
                     "normalize", "shared_normalize_param", "restrain_non_bond_pred", "reset_scale_shift",
                     "coulomb_charge_correct", "batch_norm", "uni_task_ss", "requires_atom_embedding", "lin_last",
                     "last_lin_bias", "train_shift", "stop_low_lr", "acsf", "mask_z", "time_debug", "mem_debug"]:
        assert config_dict[bool_key] is not None, f"{bool_key} is not set!"
        if isinstance(config_dict[bool_key], str):
            config_dict[bool_key] = (config_dict[bool_key].lower() != "false")
        else:
            assert isinstance(config_dict[bool_key], bool)
    if isinstance(config_dict["use_trained_model"], str):
        # super confusing, will make it clearer when possible
        config_dict["use_trained_model"] = config_dict["use_trained_model"] \
            if config_dict["use_trained_model"].lower() != "false" else False
    config_dict["use_swag"] = (config_dict["uncertainty_modify"].split('_')[0] == 'swag')
    config_dict["n_atom_embedding"] = 95

    # config_dict["n_output"] = len(config_dict["target_names"])
    # if config_dict["action"].endswith("_and_QD"):
    #     config_dict["n_output"] += 1
    return config_dict


def init_model_test(config_dict: Any, state_dict: Any) -> Any:
    from Networks.PhysDimeNet import PhysDimeNet
    model = PhysDimeNet(**preprocess_config(config_dict)).to(get_device()).type(floating_type)
    model = AveragedModel(model)
    if "n_averaged" not in state_dict.keys():
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def non_collapsing_folder(folder_prefix: str, identify: Any="_run_") -> Any:
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S__%f')
        run_directory = folder_prefix + identify + current_time
        time.sleep(random.Random().random() * 3)
        if not os.path.exists(run_directory):
            # add randomness to avoid collapsing
            os.makedirs(run_directory, exist_ok=False)
            return run_directory
        else:
            rd_sleep = random.Random().random() * 20
            print(f"Folder exists, trying to wait {rd_sleep} seconds")
            time.sleep(rd_sleep)


def add_parser_arguments(parser: Any) -> Any:
    """
    add arguments to parser
    :param parser:
    :return: added parser
    """
    parser.add_argument('--debug_mode', type=str, default="False")
    parser.add_argument('--modules', type=str, help="eg: D P D P D P, D for DimeNet and P for PhysNet")
    parser.add_argument('--bonding_type', type=str, help="eg: B N B N B N, B for bonding-edge, N for non-bonding "
                                                         "edge. (future) L for long-range interaction and BN for both "
                                                         "bonding and non-bonding")
    parser.add_argument('--n_feature', type=int)
    parser.add_argument('--n_dime_before_residual', type=int)
    parser.add_argument('--n_dime_after_residual', type=int)
    parser.add_argument('--n_output_dense', type=int)
    parser.add_argument('--n_phys_atomic_res', type=int)
    parser.add_argument('--n_phys_interaction_res', type=int)
    parser.add_argument('--n_phys_output_res', type=int)
    parser.add_argument('--n_bi_linear', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--ema_decay', type=float, help='Deprecated, use --optimizer option instead')
    parser.add_argument('--l2lambda', type=float)
    parser.add_argument('--nh_lambda', type=float)
    parser.add_argument('--decay_steps', type=int)
    parser.add_argument('--decay_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--valid_batch_size', type=int)
    parser.add_argument('--force_weight', type=float)
    parser.add_argument('--charge_weight', type=float)
    parser.add_argument('--dipole_weight', type=float)
    parser.add_argument('--use_trained_model', type=str)
    parser.add_argument('--max_norm', type=float)
    parser.add_argument('--log_file_name', type=str)
    parser.add_argument('--folder_prefix', type=str)
    parser.add_argument('--config_name', type=str, default='config.txt')
    parser.add_argument('--normalize', type=str)
    parser.add_argument('--shared_normalize_param', type=str)
    parser.add_argument('--edge_version', type=str, help="voronoi | cutoff")
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--boundary_factor', type=float)
    parser.add_argument('--activations', type=str, help='swish | ssp')
    parser.add_argument('--expansion_fn', type=str)
    parser.add_argument('--restrain_non_bond_pred', type=str)
    parser.add_argument('--frag9_train_size', type=int, help="solely used for training curve")
    parser.add_argument('--frag20_train_size', type=int, help="solely used for training curve")
    parser.add_argument('--test_interval', type=str, help="DONT USE! For compatibility only, no longer used.")
    parser.add_argument('--warm_up_steps', type=int, help="Steps to warm up")
    parser.add_argument('--data_provider', type=str, help="Data provider arguments:"
                                                          " qm9 | frag9to20_jianing | frag9to20_all")
    parser.add_argument('--data_root', type=str, default="../dataProviders/data")
    parser.add_argument('--uncertainty_modify', type=str, default='none',
                        help="none | concreteDropoutModule | concreteDropoutOutput | swag_${start}_${freq}")
    parser.add_argument('--early_stop', type=int, default=-1, help="early stopping, set to -1 to disable")
    parser.add_argument('--optimizer', type=str, default='emaAms_0.999', help="emaAms_${ema} | sgd")
    parser.add_argument('--scheduler', type=str, default='StepLR', help="StepLR | ReduceLROnPlateau")
    parser.add_argument('--freeze_option', type=str, default='none', help='none | prev | prev_extra')
    parser.add_argument('--comment', type=str, help='just comment')
    parser.add_argument('--remove_atom_ids', type=int, default=[], action="append", help='remove atoms from dataset')
    parser.add_argument('--coulomb_charge_correct', type=str, default="False",
                        help='calculate charge correction when calculation Coulomb interaction')
    parser.add_argument('--reset_optimizer', type=str, default="True",
                        help='If true, will reset optimizer/scheduler regardless of if you use pretrained model or not')
    parser.add_argument("--action", type=str, default="E", help="name of target, must be consistent with name in"
                                                                "data_provider, default E is for PhysNet energy")
    parser.add_argument("--target_names", type=str, action="append", default=[],
                        help="For Frag20-solvation: gasEnergy | watEnergy | octEnergy | CalcSol | OctSol")
    parser.add_argument("--auto_sol", type=str, default="False",
                        help="Automatic calculate solvation energy by subtracting solvent energy by gas energy.")
    parser.add_argument("--target_nodes", type=str, default="False",
                        help="Add extra nodes (fake atoms) for each target, the result of each target will be the "
                             "aggregated repr of each node.")
    parser.add_argument("--reset_output_layers", type=str, default="False")
    parser.add_argument("--reset_scale_shift", type=str, default="False")
    parser.add_argument("--loss_metric", type=str, default="mae", help="mae|rmse|mse|ce|evidential")
    parser.add_argument("--pooling", type=str, default="sum",
                        help="sum | mem_pooling[heads=?,num_clusters=?,tau=?,n_output=?]")
    parser.add_argument("--batch_norm", type=str, default="False")
    parser.add_argument("--dropout", type=str, default="False")
    parser.add_argument("--uni_task_ss", type=str, default="False", help="Universal scale/shift for all tasks.")
    parser.add_argument("--requires_atom_embedding", type=str, default="False")
    parser.add_argument("--lin_last", type=str, default="False", help="vi: scale, shift -> sum -> lin")
    parser.add_argument("--last_lin_bias", type=str, default="False")
    parser.add_argument("--stop_low_lr", type=str, default="False")
    parser.add_argument("--acsf", type=str, default="False")
    parser.add_argument("--train_shift", type=str, default="True")
    parser.add_argument("--mask_z", type=str, default="False")
    parser.add_argument("--z_loss_weight", type=float, default=0)
    parser.add_argument("--mask_atom", action="store_true", help="Only predict part of the atomic properties")
    parser.add_argument("--time_debug", type=str, default="False")
    parser.add_argument("--mem_debug", type=str, default="False")
    parser.add_argument("--chk", type=str, default=None)
    parser.add_argument("--n_output", type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--keep", type=str, default=None)
    parser.add_argument("--ext_atom_features", type=str, default=None)
    parser.add_argument("--flex_sol", action="store_true",
                        help="Multi-task FT on experimental datasets: use MT when available, otherwise use st")
    parser.add_argument("--over_sample", action="store_true")
    return parser


# updated evidential regression loss
def evidential_loss_new(mu: Any, v: Any, alpha: Any, beta: Any, targets: Any, lam: Any=1, epsilon: Any=1e-4) -> Any:
    """
    Adapted from https://pubs.acs.org/doi/10.1021/acscentsci.1c00546
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    L_NLL = nll

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss


def remove_handler(log: Any=None) -> Any:
    if log is None:
        log = logging.getLogger()
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    return


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


def process_state_dict(state_dict: Any, config_dict: Any, logger: Any, is_main: Any) -> Any:
    if config_dict["reset_output_layers"] or config_dict["reset_scale_shift"]:
        # OrderedDict is immutable so I have to make a copy
        new_state_dict = OrderedDict()
        shift_reg = re.compile(r"shift.*")
        scale_reg = re.compile(r"scale.*")
        output_reg = re.compile(r"main_module_list.*\.output\.lin\..*")

        reset_list = []
        if config_dict["reset_output_layers"]:
            logger.info("reset output layers...")
            reset_list.append(output_reg)
        if config_dict["reset_scale_shift"]:
            logger.info("reset scale and shift...")
            reset_list.append(shift_reg)
            reset_list.append(scale_reg)
        for key in state_dict:
            keep = True
            for reg in reset_list:
                if reg.fullmatch(key) is not None:
                    keep = False
                    if is_main:
                        logger.info(f"discarding: {key}")
                    break
            if keep:
                new_state_dict[key] = state_dict[key]
        return new_state_dict
    else:
        return state_dict


def validate_index(train_index: Any, val_index: Any, test_index: Any) -> Any:
    # make sure the indexes are legit without overlapping, etc...
    train_size = train_index.shape[0]
    train_index_set = set(train_index.numpy().tolist())
    assert train_size == len(train_index_set), f"{train_size}, {len(train_index_set)}"

    val_size = val_index.shape[0]
    val_index_set = set(val_index.numpy().tolist())
    assert val_size == len(val_index_set), f"{val_size}, {len(val_index_set)}"
    assert len(train_index_set.intersection(val_index_set)) == 0, "You have a problem :)"

    if test_index is not None:
        test_size = test_index.shape[0]
        test_index_set = set(test_index.numpy().tolist())
        assert test_size == len(test_index_set), f"{test_size}, {len(test_index_set)}"
        assert len(train_index_set.intersection(test_index_set)) == 0, "You have a problem :)"
    else:
        test_size = None

    return train_size, val_size, test_size

def solv_num_workers() -> Any:
    try:
        n_cpu_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu_avail = None
    n_cpu = os.cpu_count()
    num_workers = n_cpu_avail if n_cpu_avail is not None else n_cpu
    return n_cpu_avail, n_cpu, num_workers


if __name__ == '__main__':
    dummy_input = torch.rand(32, 160).double().cuda()
    dummy_model = torch.nn.Linear(160, 4).double().cuda()
    dummy_input = dummy_model(dummy_input)
    means, log_lambdas, log_alphas, log_betas = torch.split(dummy_input, dummy_input.shape[-1] // 4, dim=-1)
    soft_plus = torch.nn.Softplus()
    min_val = 1e-6
    lambdas = soft_plus(log_lambdas) + min_val
    # add 1 for numerical contraints of Gamma function
    alphas = soft_plus(log_alphas) + min_val + 1
    betas = soft_plus(log_betas) + min_val
    evi_cal_dict = {"mu": means, "v": lambdas, "alpha": alphas, "beta": betas}
    prop_pred = means
    evi_cal_dict["targets"] = torch.rand(32, 1).double().cuda()
    loss = evidential_loss_new(**evi_cal_dict).sum()
    loss.backward()
    print("finished")
