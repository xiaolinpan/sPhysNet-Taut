from typing import Any
import logging
import time
from math import ceil

import torch

from taut_src.Networks.SharedLayers.ResidualLayer import ResidualLayer
from taut_src.Networks.SharedLayers.ActivationFns import activation_getter
from taut_src.Networks.PhysLayers.Interaction_module import InteractionModule
from taut_src.Networks.UncertaintyLayers.MCDropout import ConcreteDropout
from taut_src.utils.time_meta import record_data
from taut_src.utils.utils_functions import floating_type, get_n_params, option_solver, _get_index_from_matrix


class OutputLayer(torch.nn.Module):
    """
    The output layer(red one in paper) of PhysNet
    """

    def __init__(self, F: Any, n_output: Any, n_res_output: Any, activation: Any, uncertainty_modify: Any, n_read_out: Any=0, batch_norm: Any=False,
                 dropout: Any=False, zero_last_linear: Any=True, bias: Any=False) -> None:
        self.batch_norm = batch_norm
        super().__init__()
        self.concrete_dropout = (uncertainty_modify.split('[')[0] == "concreteDropoutOutput")
        self.dropout_options = option_solver(uncertainty_modify)
        # convert string into correct types:
        if 'train_p' in self.dropout_options:
            self.dropout_options['train_p'] = (self.dropout_options['train_p'].lower() == 'true')
        if 'normal_dropout' in self.dropout_options:
            self.dropout_options['normal_dropout'] = (self.dropout_options['normal_dropout'].lower() == 'true')
        if 'init_min' in self.dropout_options:
            self.dropout_options['init_min'] = float(self.dropout_options['init_min'])
        if 'init_max' in self.dropout_options:
            self.dropout_options['init_max'] = float(self.dropout_options['init_max'])

        self.n_res_output = n_res_output
        self.n_read_out = n_read_out
        for i in range(n_res_output):
            self.add_module('res_layer' + str(i), ResidualLayer(F, activation, concrete_dropout=False,
                                                                batch_norm=batch_norm, dropout=dropout))

        # Readout layers
        dim_decay = True  # this is for compatibility issues, always set to True otherwise
        if not dim_decay:
            print('WARNING, dim decay is not enabled!')
        last_dim = F
        for i in range(n_read_out):
            if dim_decay:
                this_dim = ceil(last_dim/2)
                read_out_i = torch.nn.Linear(last_dim, this_dim)
                last_dim = this_dim
            else:
                read_out_i = torch.nn.Linear(last_dim, last_dim)
                this_dim = last_dim
            if self.concrete_dropout:
                read_out_i = ConcreteDropout(read_out_i, module_type='Linear', **self.dropout_options)
            self.add_module('read_out{}'.format(i), read_out_i)
            if self.batch_norm:
                self.add_module("bn_{}".format(i), torch.nn.BatchNorm1d(last_dim, momentum=1.))

        self.lin = torch.nn.Linear(last_dim, n_output, bias=bias)
        if zero_last_linear:
            self.lin.weight.data.zero_()
        else:
            logging.info("Output layer not zeroed, make sure you are doing classification.")
        if self.concrete_dropout:
            self.lin = ConcreteDropout(self.lin, module_type='Linear', **self.dropout_options)
        if self.batch_norm:
            self.bn_last = torch.nn.BatchNorm1d(last_dim, momentum=1.)

        self.activation = activation_getter(activation)

    def forward(self, x: Any) -> Any:
        tmp_res = x
        regularization = 0.

        for i in range(self.n_res_output):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res)
        out = tmp_res

        for i in range(self.n_read_out):
            if self.batch_norm:
                out = self._modules["bn_{}".format(i)](out)
            a = self.activation(out)
            out = self._modules['read_out{}'.format(i)](a)
            if self.concrete_dropout:
                regularization = regularization + out[1]
                out = out[0]

        if self.batch_norm:
            out = self.bn_last(out)
        embed_b4_ss = self.activation(out)
        out = embed_b4_ss
        if self.concrete_dropout:
            out, reg = self.lin(out)
            regularization = regularization + reg
        else:
            out = self.lin(out)
        return out, regularization, embed_b4_ss

    def freeze_residual_layers(self) -> Any:
        for i in range(self.n_res_output):
            for param in getattr(self, f"res_layer{i}").parameters():
                param.requires_grad_(False)


class PhysModule(torch.nn.Module):
    """
    Main module in PhysNet
    """

    def __init__(self, F: Any, K: Any, n_output: Any, n_res_atomic: Any, n_res_interaction: Any, n_res_output: Any, activation: Any, uncertainty_modify: Any,
                 n_read_out: Any, batch_norm: Any, dropout: Any, zero_last_linear: Any, bias: Any) -> None:
        super().__init__()
        self.interaction = InteractionModule(F=F, K=K, n_res_interaction=n_res_interaction, activation=activation,
                                             batch_norm=batch_norm, dropout=dropout).type(floating_type)
        self.n_res_atomic = n_res_atomic
        for i in range(n_res_atomic):
            self.add_module('res_layer' + str(i), ResidualLayer(F, activation, batch_norm=batch_norm, dropout=dropout))
        self.output = OutputLayer(F=F, n_output=n_output, n_res_output=n_res_output, activation=activation,
                                  uncertainty_modify=uncertainty_modify, n_read_out=n_read_out, batch_norm=batch_norm,
                                  dropout=dropout, zero_last_linear=zero_last_linear, bias=bias)

    def forward(self, input_dict: Any) -> Any:
        # t0 = time.time()

        x = input_dict["vi"]
        edge_index = input_dict["edge_index"]
        edge_attr = input_dict["edge_attr"]["rbf"]

        # t0 = record_data('assign values', t0)
        interacted_x, _ = self.interaction(x, edge_index, edge_attr)

        # t0 = record_data('interaction layer', t0)

        tmp_res = interacted_x
        for i in range(self.n_res_atomic):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res)

        # t0 = record_data('residual layer', t0)

        # embedding before shift and scale
        out_res, regularization, embed_b4_ss = self.output(tmp_res)

        # t0 = record_data('output layer', t0)

        return {"vi": tmp_res, "out": out_res, "regularization": regularization, "embed_b4_ss": embed_b4_ss}

    def freeze_prev_layers(self) -> Any:
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.output.parameters():
            param.requires_grad_(True)
        return


if __name__ == '__main__':
    pass
