import logging
import math
import time

import torch
import torch.nn as nn
from torch_scatter import scatter

from utils.DataPrepareUtils import cal_msg_edge_index, extend_bond
from Networks.DimeLayers.DimeModule import DimeModule
from Networks.PhysLayers.CoulombLayer import CoulombLayer
from Networks.PhysLayers.D3DispersionLayer import D3DispersionLayer
from Networks.PhysLayers.PhysModule import PhysModule
from Networks.SharedLayers.AtomToEdgeLayer import AtomToEdgeLayer
from Networks.SharedLayers.EdgeToAtomLayer import EdgeToAtomLayer
from Networks.SharedLayers.EmbeddingLayer import EmbeddingLayer
from Networks.SharedLayers.MyMemPooling import MyMemPooling
from Networks.UncertaintyLayers.MCDropout import ConcreteDropout
from utils.BesselCalculator import bessel_expansion_raw
from utils.BesselCalculatorFast import BesselCalculator
from utils.tags import tags
from utils.time_meta import record_data
from utils.utils_functions import floating_type, dime_edge_expansion, softplus_inverse, \
    gaussian_rbf, info_resolver, expansion_splitter, error_message, option_solver, get_device


class PhysDimeNet(nn.Module):
    """
    Combination of PhyNet and DimeNet
    For non-bonding interaction, using atom-atom interaction in PhysNet
    For bonding interaction, using directional message passing
    Final prediction is the combination of PhysNet and DimeNet

    Next time I will use **kwargs :<
    """

    def __init__(self,
                 n_atom_embedding,
                 modules,
                 bonding_type,
                 n_feature,
                 n_output,
                 n_dime_before_residual,
                 n_dime_after_residual,
                 n_output_dense,
                 n_phys_atomic_res,
                 n_phys_interaction_res,
                 n_phys_output_res,
                 n_bi_linear,
                 nh_lambda,
                 normalize,
                 shared_normalize_param,
                 activations,
                 restrain_non_bond_pred,
                 expansion_fn,
                 uncertainty_modify,
                 coulomb_charge_correct,
                 loss_metric,
                 uni_task_ss,
                 lin_last,
                 last_lin_bias,
                 train_shift,
                 mask_z,
                 time_debug,
                 z_loss_weight,
                 acsf=False,
                 energy_shift=None,
                 energy_scale=None,
                 debug_mode=False,
                 action="E",
                 target_names=None,
                 batch_norm=False,
                 dropout=False,
                 requires_atom_prop=False,
                 requires_atom_embedding=False,
                 pooling="sum",
                 ext_atom_features=None,
                 ext_atom_dim=0,
                 **kwargs):
        """
        
        :param n_atom_embedding: number of atoms to embed, usually set to 95
        :param n_feature:
        :param n_output: 
        :param n_dime_before_residual:
        :param n_dime_after_residual:
        :param n_output_dense:
        :param n_bi_linear: Dimension of bi-linear layer in DimeNet, also called N_tensor in paper
        :param nh_lambda: non-hierarchical penalty coefficient
        :param debug_mode: 
        """
        super().__init__()
        self.ext_atom_dim = ext_atom_dim
        self.ext_atom_features = ext_atom_features
        self.z_loss_weight = z_loss_weight
        if z_loss_weight > 0:
            assert mask_z, "If you want to predict Z (atomic number), please mask Z in input"
        self.mask_z = mask_z
        self.time_debug = time_debug
        self.use_acsf = acsf
        if acsf:
            self.acsf_convert = nn.Linear(260, n_feature)
        self.train_shift = train_shift
        self.last_lin_bias = last_lin_bias
        self.lin_last = lin_last
        self.requires_atom_embedding = requires_atom_embedding
        self.uni_task_ss = uni_task_ss
        self.loss_metric = loss_metric
        self.pooling = pooling

        self.logger = logging.getLogger()
        self.requires_atom_prop = requires_atom_prop
        if action in tags.requires_atomic_prop:
            self.logger.info("Overwriting self.requires_atom_prop = True because you require atomic props")
            self.requires_atom_prop = True
        # convert input into a dictionary
        self.target_names = target_names
        self.num_targets = len(target_names)
        self.action = action
        self.coulomb_charge_correct = coulomb_charge_correct
        self.uncertainty_modify = uncertainty_modify
        self.expansion_fn = expansion_splitter(expansion_fn)

        self.activations = activations.split(' ')
        module_str_list = modules.split()
        bonding_str_list = bonding_type.split()
        # main modules, including P (PhysNet), D (DimeNet), etc.
        self.main_module_str = []
        self.main_bonding_str = []
        # post modules are either C (Coulomb) or D3 (D3 Dispersion)
        self.post_module_str = []
        self.post_bonding_str = []
        for this_module_str, this_bonding_str in zip(module_str_list, bonding_str_list):
            '''
            Separating main module and post module
            '''
            this_module_str = this_module_str.split('[')[0]
            if this_module_str in ['D', 'P', 'D-noOut', 'P-noOut']:
                self.main_module_str.append(this_module_str)
                self.main_bonding_str.append(this_bonding_str)
            elif this_module_str in ['C', 'D3']:
                self.post_module_str.append(this_module_str)
                self.post_bonding_str.append(this_bonding_str)
            else:
                error_message(this_module_str, 'module')
        self.bonding_type_keys = set(bonding_str_list)

        # whether calculate long range interaction or not
        self.contains_lr = False
        for _bonding_type in self.bonding_type_keys:
            if _bonding_type.find('L') >= 0:
                self.contains_lr = True

        module_bond_combine_str = []
        msg_bond_type = []
        for module, bond in zip(module_str_list, bonding_str_list):
            module = module.split('[')[0]
            module_bond_combine_str.append('{}_{}'.format(module, bond))
            if module == 'D':
                msg_bond_type.append(bond)
        self.module_bond_combine_str = set(module_bond_combine_str)
        # These bonds need to be expanded into message version (i.e. 3 body interaction)
        self.msg_bond_type = set(msg_bond_type)

        self.restrain_non_bond_pred = restrain_non_bond_pred
        self.shared_normalize_param = shared_normalize_param
        self.normalize = normalize
        self.n_phys_output_res = n_phys_output_res
        self.n_phys_interaction_res = n_phys_interaction_res
        self.n_phys_atomic_res = n_phys_atomic_res
        self.debug_mode = debug_mode
        if n_output == 0:
            # Here we setup a fake n_output to avoid errors in initialization
            # But the sum pooling result will not be used
            self.no_sum_output = True
            n_output = 1
        else:
            self.no_sum_output = False
        self.n_output = n_output
        self.nhlambda = nh_lambda

        # A dictionary which parses an expansion combination into detailed information
        self.expansion_info_getter = {
            combination: info_resolver(self.expansion_fn[combination])
            for combination in self.module_bond_combine_str
        }

        # registering necessary parameters for some expansions if needed
        for combination in self.expansion_fn.keys():
            expansion_fn_info = self.expansion_info_getter[combination]
            if expansion_fn_info['name'] == "gaussian":
                n_rbf = expansion_fn_info['n']
                feature_dist = expansion_fn_info['dist']
                feature_dist = torch.as_tensor(feature_dist).type(floating_type)
                self.register_parameter('cutoff' + combination, torch.nn.Parameter(feature_dist, False))
                # Centers are params for Gaussian RBF expansion in PhysNet
                centers = softplus_inverse(torch.linspace(1.0, math.exp(-feature_dist), n_rbf))
                centers = torch.nn.functional.softplus(centers)
                self.register_parameter('centers' + combination, torch.nn.Parameter(centers, True))

                # Widths are params for Gaussian RBF expansion in PhysNet
                widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-feature_dist)) / n_rbf)) ** 2)] * n_rbf
                widths = torch.as_tensor(widths).type(floating_type)
                widths = torch.nn.functional.softplus(widths)
                self.register_parameter('widths' + combination, torch.nn.Parameter(widths, True))
            elif expansion_fn_info['name'] == 'defaultDime':
                n_srbf = self.expansion_info_getter[combination]['n_srbf']
                n_shbf = self.expansion_info_getter[combination]['n_shbf']
                envelop_p = self.expansion_info_getter[combination]['envelop_p']
                setattr(self, f"bessel_calculator_{n_srbf}_{n_shbf}", BesselCalculator(n_srbf, n_shbf, envelop_p))

        self.base_unit_getter = {
            'D': "edge",
            'D-noOut': "edge",
            'P': "atom",
            'P-noOut': "atom",
            'C': "atom",
            'D3': "atom"
        }

        self.dist_calculator = nn.PairwiseDistance(keepdim=True)

        self.embedding_layer = EmbeddingLayer(n_atom_embedding, n_feature-self.ext_atom_dim)

        previous_module = 'P'
        '''
        registering main modules
        '''
        self.main_module_list = nn.ModuleList()
        # stores extra info including module type, bonding type, etc
        self.main_module_info = []
        for i, (module_str, bonding_str) in enumerate(zip(module_str_list, bonding_str_list)):
            # contents within '[]' will be considered options
            _options = option_solver(module_str)
            module_str = module_str.split('[')[0]
            combination = module_str + "_" + bonding_str
            if module_str in ['D', 'D-noOut']:
                n_dime_rbf = self.expansion_info_getter[combination]['n']
                n_srbf = self.expansion_info_getter[combination]['n_srbf']
                n_shbf = self.expansion_info_getter[combination]['n_shbf']
                dim_sbf = n_srbf * (n_shbf + 1)
                this_module_str = DimeModule(dim_rbf=n_dime_rbf,
                                             dim_sbf=dim_sbf,
                                             dim_msg=n_feature,
                                             n_output=n_output,
                                             n_res_interaction=n_dime_before_residual,
                                             n_res_msg=n_dime_after_residual,
                                             n_dense_output=n_output_dense,
                                             dim_bi_linear=n_bi_linear,
                                             activation=self.activations[i],
                                             uncertainty_modify=uncertainty_modify)
                if self.uncertainty_modify == 'concreteDropoutModule':
                    this_module_str = ConcreteDropout(this_module_str, module_type='DimeNet')
                self.main_module_list.append(this_module_str)
                self.main_module_info.append({"module_str": module_str, "bonding_str": bonding_str,
                                              "combine_str": "{}_{}".format(module_str, bonding_str),
                                              "is_transition": False})
                if self.base_unit_getter[previous_module] == "atom":
                    self.main_module_list.append(AtomToEdgeLayer(n_dime_rbf, n_feature, self.activations[i]))
                    self.main_module_info.append({"is_transition": True})

            elif module_str in ['P', 'P-noOut']:
                this_module_str = PhysModule(F=n_feature,
                                             K=self.expansion_info_getter[combination]['n'],
                                             n_output=n_output,
                                             n_res_atomic=n_phys_atomic_res,
                                             n_res_interaction=n_phys_interaction_res,
                                             n_res_output=n_phys_output_res,
                                             activation=self.activations[i],
                                             uncertainty_modify=uncertainty_modify,
                                             n_read_out=int(_options['n_read_out']) if 'n_read_out' in _options else 0,
                                             batch_norm=batch_norm,
                                             dropout=dropout,
                                             zero_last_linear=(loss_metric != "ce"),
                                             bias=last_lin_bias)
                if self.uncertainty_modify == 'concreteDropoutModule':
                    this_module_str = ConcreteDropout(this_module_str, module_type='PhysNet')
                self.main_module_list.append(this_module_str)
                self.main_module_info.append({"module_str": module_str, "bonding_str": bonding_str,
                                              "combine_str": "{}_{}".format(module_str, bonding_str),
                                              "is_transition": False})
                if self.base_unit_getter[previous_module] == "edge":
                    self.main_module_list.append(EdgeToAtomLayer())
                    self.main_module_info.append({"is_transition": True})
            elif module_str in ['C', 'D3']:
                pass
            else:
                error_message(module_str, 'module')
            previous_module = module_str

        # TODO Post modules to list
        for i, (module_str, bonding_str) in enumerate(zip(self.post_module_str, self.post_bonding_str)):
            if module_str == 'C':
                combination = module_str + "_" + bonding_str
                self.add_module('post_module{}'.format(i),
                                CoulombLayer(cutoff=self.expansion_info_getter[combination]['dist']))
            elif module_str == 'D3':
                self.add_module('post_module{}'.format(i), D3DispersionLayer(s6=0.5, s8=0.2130, a1=0.0, a2=6.0519))
            else:
                error_message(module_str, 'module')

        # TODO normalize to list
        if self.normalize:
            '''
            Atom-wise shift and scale, used in PhysNet
            '''
            if uni_task_ss:
                ss_dim = 1
            else:
                ss_dim = n_output
            if shared_normalize_param:
                shift_matrix = torch.zeros(95, ss_dim).type(floating_type)
                scale_matrix = torch.zeros(95, ss_dim).type(floating_type).fill_(1.0)
                if energy_shift is not None:
                    if isinstance(energy_shift, torch.Tensor):
                        shift_matrix[:, :] = energy_shift.view(1, -1)[:, :ss_dim]
                    else:
                        shift_matrix[:, 0] = energy_shift
                if energy_scale is not None:
                    if isinstance(energy_scale, torch.Tensor):
                        scale_matrix[:, :] = energy_scale.view(1, -1)[:, :ss_dim]
                    else:
                        scale_matrix[:, 0] = energy_scale
                shift_matrix = shift_matrix / len(self.bonding_type_keys)
                self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
                self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=train_shift))
            else:
                for key in self.bonding_type_keys:
                    shift_matrix = torch.zeros(95, ss_dim).type(floating_type)
                    scale_matrix = torch.zeros(95, ss_dim).type(floating_type).fill_(1.0)
                    if energy_shift is not None:
                        shift_matrix[:, 0] = energy_shift
                    if energy_scale is not None:
                        scale_matrix[:, 0] = energy_scale
                    shift_matrix = shift_matrix / len(self.bonding_type_keys)
                    self.register_parameter('scale{}'.format(key), torch.nn.Parameter(scale_matrix, requires_grad=True))
                    self.register_parameter('shift{}'.format(key),
                                            torch.nn.Parameter(shift_matrix, requires_grad=train_shift))

        pool_base, pool_options = option_solver(pooling, type_conversion=True, return_base=True)
        if pool_base == "sum":
            self.pooling_module = None
        elif pool_base == "mem_pooling":
            self.pooling_module = MyMemPooling(in_channels=n_feature, **pool_options)
        else:
            raise ValueError(f"invalid pool_base: {pool_base}")

        if self.lin_last:
            # extra safety
            assert self.requires_atom_embedding
            assert isinstance(self.main_module_list[-1], PhysModule)

    def freeze_prev_layers(self, freeze_extra=False):
        if freeze_extra:
            # Freeze scale, shift and Gaussian RBF parameters
            for param in self.parameters():
                param.requires_grad_(False)
        for i in range(len(self.main_module_str)):
            self.main_module_list[i].freeze_prev_layers()
            self.main_module_list[i].output.freeze_residual_layers()

    def forward(self, data):
        # torch.cuda.synchronize(device=device)
        t0 = time.time()

        R = data.pos.type(floating_type)
        Z = data.Z
        if self.mask_z:
            Z = torch.zeros_like(Z)
        N = data.N
        '''
        Note: non_bond_edge_index is for non-bonding interactions
              bond_edge_index is for bonding interactions
        '''
        atom_mol_batch = data.batch
        edge_index_getter = {}
        # we now support bonding type separated by '.'
        for bonding_str in self.bonding_type_keys:
            # eg: B.N -> BN
            bonding_str_raw = "".join(bonding_str.split('.'))
            edge_index = getattr(data, bonding_str_raw + '_edge_index', False)
            if edge_index is not False:
                # edge_index_getter[bonding_str] = edge_index + getattr(data, bonding_str_raw + '_edge_index_correct')
                edge_index_getter[bonding_str] = edge_index
            # elif len(bonding_str) > 4 and bonding_str[-4:] == "-ext":
            #     # calculate 1,3 interaction on the fly
            #     b4_ext = bonding_str.split('-')[0]
            #     if b4_ext in edge_index_getter.keys():
            #         b4_ext_index = edge_index_getter[b4_ext]
            #     else:
            #         b4_ext_index = data[b4_ext + '_edge_index'] + data[b4_ext + '_edge_index_correct']
            #     edge_index_getter[bonding_str] = extend_bond(b4_ext_index)
            # else:
            #     edge_index_getter[bonding_str] = torch.cat(
            #         [data[_type + '_edge_index'] + data[_type + '_edge_index_correct'] for _type in bonding_str],
            #         dim=-1)


        msg_edge_index_getter = {}
        for bonding_str in self.msg_bond_type:
            # prepare msg edge index
            this_msg_edge_index = getattr(data, bonding_str + '_msg_edge_index', False)
            if this_msg_edge_index is not False:
                msg_edge_index_getter[bonding_str] = this_msg_edge_index + \
                                                     getattr(data, bonding_str + '_msg_edge_index_correct')
            else:
                msg_edge_index_getter[bonding_str] = cal_msg_edge_index(edge_index_getter[bonding_str]).to(get_device())

        if self.time_debug:
            t0 = record_data("msg_bond_setup", t0)

        expansions = {}
        '''
        calculating expansion
        '''
        for combination in self.module_bond_combine_str:
            module_str = combination.split('_')[0]
            this_bond = combination.split('_')[1]
            this_expansion = self.expansion_info_getter[combination]['name']
            if module_str in ['D', 'D-noOut']:
                # DimeNet, calculate sbf and rbf
                if this_expansion == "defaultDime":
                    n_srbf = self.expansion_info_getter[combination]['n_srbf']
                    n_shbf = self.expansion_info_getter[combination]['n_shbf']
                    expansions[combination] = dime_edge_expansion(R, edge_index_getter[this_bond],
                                                                  msg_edge_index_getter[this_bond],
                                                                  self.expansion_info_getter[combination]['n'],
                                                                  self.dist_calculator,
                                                                  getattr(self, f"bessel_calculator_{n_srbf}_{n_shbf}"),
                                                                  self.expansion_info_getter[combination]['dist'],
                                                                  return_dict=True)
                else:
                    raise ValueError("Double check your expansion input!")
            elif module_str in ['P', 'P-noOut']:
                # PhysNet, calculate rbf
                if this_expansion == 'bessel':
                    this_edge_index = edge_index_getter[this_bond]
                    dist_atom = self.dist_calculator(R[this_edge_index[0, :], :], R[this_edge_index[1, :], :])
                    rbf = bessel_expansion_raw(dist_atom, self.expansion_info_getter[combination]['n'],
                                               self.expansion_info_getter[combination]['dist'])
                    expansions[combination] = {"rbf": rbf}
                elif this_expansion == 'gaussian':
                    this_edge_index = edge_index_getter[this_bond]
                    pair_dist = self.dist_calculator(R[this_edge_index[0, :], :], R[this_edge_index[1, :], :])
                    expansions[combination] = gaussian_rbf(pair_dist, getattr(self, 'centers' + combination),
                                                           getattr(self, 'widths' + combination),
                                                           getattr(self, 'cutoff' + combination),
                                                           return_dict=True)
                else:
                    error_message(this_expansion, 'expansion')
            elif (module_str == 'C') or (module_str == 'D3'):
                '''
                In this situation, we only need to calculate pair-wise distance.
                '''
                this_edge_index = edge_index_getter[this_bond]
                # TODO pair dist was calculated twice here
                expansions[combination] = {"pair_dist": self.dist_calculator(R[this_edge_index[0, :], :],
                                                                             R[this_edge_index[1, :], :])}
            else:
                # something went wrong
                error_message(module_str, 'module')

        if self.time_debug:
            t0 = record_data("expansion_prepare", t0)

        separated_last_out = {key: None for key in self.bonding_type_keys}
        separated_out_sum = {key: 0. for key in self.bonding_type_keys}

        nh_loss = torch.zeros(1).type(floating_type).to(get_device())
        mji = None
        '''
        mji: edge diff
        vi:  node diff
        '''
        if self.use_acsf:
            vi_init = data.acsf.type(floating_type)
            vi_init = self.acsf_convert(vi_init)
        else:
            vi_init = self.embedding_layer(Z)
            if self.ext_atom_features is not None:
                vi_init = torch.cat([vi_init, getattr(data, self.ext_atom_features)], dim=-1)
        out_dict = {"vi": vi_init, "mji": mji}

        if self.time_debug:
            t0 = record_data("embedding_prepare", t0)

        output = {}
        '''
        Going through main modules
        '''
        for i, (info, _module) in enumerate(zip(self.main_module_info, self.main_module_list)):
            out_dict["info"] = info
            if not info["is_transition"]:
                out_dict["edge_index"] = edge_index_getter[info["bonding_str"]]
                out_dict["edge_attr"] = expansions[info["combine_str"]]
            if info["module_str"].split("-")[0] == "D":
                out_dict["msg_edge_index"] = msg_edge_index_getter[info["bonding_str"]]

            out_dict = _module(out_dict)

            if self.z_loss_weight > 0 and i == 0:
                # first layer embedding will be used to predict atomic number (Z)
                output["first_layer_vi"] = out_dict["vi"]

            if info["is_transition"] or info["module_str"].split("-")[-1] == '-noOut':
                pass
            else:
                nh_loss = nh_loss + out_dict["regularization"]
                if separated_last_out[info["bonding_str"]] is not None:
                    # Calculating non-hierarchical penalty
                    out2 = out_dict["out"] ** 2
                    last_out2 = separated_last_out[info["bonding_str"]] ** 2
                    nh_loss = nh_loss + torch.mean(out2 / (out2 + last_out2 + 1e-7)) * self.nhlambda
                separated_last_out[info["bonding_str"]] = out_dict["out"]
                separated_out_sum[info["bonding_str"]] = separated_out_sum[info["bonding_str"]] + out_dict["out"]

        if self.time_debug:
            t0 = record_data("main_modules", t0)

        if not self.lin_last and self.normalize:
            '''
            Atom-wise shifting and scale
            '''
            if self.shared_normalize_param:
                for key in self.bonding_type_keys:
                    separated_out_sum[key] = self.scale[Z, :] * separated_out_sum[key] + self.shift[Z, :]
            else:
                for key in self.bonding_type_keys:
                    separated_out_sum[key] = getattr(self, 'scale{}'.format(key))[Z, :] * separated_out_sum[key] + \
                                             getattr(self, 'shift{}'.format(key))[Z, :]

        if self.time_debug:
            t0 = record_data("normalization", t0)

        atom_prop = 0.
        for key in self.bonding_type_keys:
            atom_prop = atom_prop + separated_out_sum[key]

        '''
        Post modules: Coulomb or D3 Dispersion layers
        '''
        for i, (module_str, bonding_str) in enumerate(zip(self.post_module_str, self.post_bonding_str)):
            this_edge_index = edge_index_getter[bonding_str]
            this_expansion = expansions["{}_{}".format(module_str, bonding_str)]
            if module_str == 'C':
                if self.coulomb_charge_correct:
                    Q = data.Q
                    coulomb_correction = self._modules["post_module{}".format(i)](atom_prop[:, -1],
                                                                                  this_expansion["pair_dist"],
                                                                                  this_edge_index, q_ref=Q, N=N,
                                                                                  atom_mol_batch=atom_mol_batch)
                else:
                    # print("one of the variables needed for gradient computation has been modified by an inplace"
                    #       " operation: need to be fixed here, probably in function cal_coulomb_E")
                    coulomb_correction = self._modules["post_module{}".format(i)](atom_prop[:, -1],
                                                                                  this_expansion['pair_dist'],
                                                                                  this_edge_index)
                atom_prop[:, 0] = atom_prop[:, 0] + coulomb_correction
            elif module_str == 'D3':
                d3_correction = self._modules["post_module{}".format(i)](Z, this_expansion, this_edge_index)
                atom_prop[:, 0] = atom_prop[:, 0] + d3_correction
            else:
                error_message(module_str, 'module')

        if self.time_debug:
            t0 = record_data("post_modules", t0)

        if self.restrain_non_bond_pred and ('N' in self.bonding_type_keys):
            # Bonding energy should be larger than non-bonding energy
            atom_prop2 = atom_prop ** 2
            non_bond_prop2 = separated_out_sum['N'] ** 2
            nh_loss = nh_loss + torch.mean(non_bond_prop2 / (atom_prop2 + 1e-7)) * self.nhlambda

        # Total prediction is the summation of bond and non-bond prediction
        mol_pred_properties = scatter(reduce='sum', src=atom_prop, index=atom_mol_batch, dim=0)

        if self.debug_mode:
            if torch.abs(mol_pred_properties.detach()).max() > 1e4:
                error_message(torch.abs(mol_pred_properties.detach()).max(), 'Energy prediction')

        if self.action in ["E", "names_and_QD"]:
            mol_prop = mol_pred_properties[:, :-1]
            assert self.n_output > 1
            # the last property is considered as atomic charge prediction
            Q_pred = mol_pred_properties[:, -1]
            Q_atom = atom_prop[:, -1]
            atom_prop = atom_prop[:, :-1]
            D_atom = Q_atom.view(-1, 1) * R
            D_pred = scatter(reduce='sum', src=D_atom, index=atom_mol_batch, dim=0)
            output["Q_pred"] = Q_pred
            output["D_pred"] = D_pred
            if self.requires_atom_prop:
                output["Q_atom"] = Q_atom
        else:
            mol_prop = mol_pred_properties

        output["nh_loss"] = nh_loss

        if self.pooling_module is not None:
            output["mol_prop_pool"] = self.pooling_module(atom_mol_batch=atom_mol_batch, vi=out_dict["vi"])

        if self.requires_atom_prop:
            output["atom_prop"] = atom_prop
            output["atom_mol_batch"] = atom_mol_batch

        if self.requires_atom_embedding:
            if self.normalize:
                dim_ss = self.scale.shape[-1]
                output["atom_embedding"] = torch.cat([out_dict["embed_b4_ss"] * self.scale[Z, i].reshape(-1, 1)
                                                      for i in range(dim_ss)], dim=-1)
                output["atom_embedding_ss"] = torch.cat([out_dict["embed_b4_ss"] * self.scale[Z, i].reshape(-1, 1)
                                                         + self.shift[Z, i].reshape(-1, 1)
                                                         for i in range(dim_ss)], dim=-1)
            else:
                output["atom_embedding"] = out_dict["embed_b4_ss"]
                output["atom_embedding_ss"] = out_dict["embed_b4_ss"]

        if self.lin_last:
            # took me hours to find this bug.
            assert self.uni_task_ss
            mol_embedding = scatter(output["atom_embedding"], atom_mol_batch, dim=0, reduce="sum")
            result = self.main_module_list[-1].output.lin(mol_embedding)
            if self.normalize and not self.train_shift:
                # we want to add "shift" directly onto atom reference, which is essentially atom_mean * num_atoms
                if self.loss_metric != "evidential":
                    result = result + N.view(-1, 1) * self.shift[0, :]
                else:
                    shift_correct = N.view(-1, 1) * self.shift[0, :]
                    assert self.num_targets * 4 == result.shape[-1]
                    result[:, :self.num_targets] = result[:, :self.num_targets] + shift_correct
            if self.action == "names_and_QD":
                result = result[:, :-1]
            output["mol_prop"] = result
        elif self.no_sum_output:
            output["mol_prop"] = torch.Tensor(torch.Size([mol_prop.shape[0], 0])).to(mol_pred_properties.device)
        else:
            output["mol_prop"] = mol_prop

        if self.time_debug:
            t0 = record_data("scatter_pool_others", t0)

        return output
