import argparse
import copy
import glob
import logging
import os
import os.path as osp
from datetime import datetime

import torch

from utils.DataPrepareUtils import my_pre_transform
from Networks.PhysDimeNet import PhysDimeNet
from Networks.UncertaintyLayers.swag import SWAG
from train import data_provider_solver, _add_arg_from_config, remove_extra_keys
from utils.LossFn import LossFn
from utils.utils_functions import remove_handler, get_device, floating_type, init_model_test, validate_index, \
    add_parser_arguments, preprocess_config


class TrainedFolder:
    """
    Load a trained folder for performance evaluation and other purposes
    """
    def __init__(self, folder_name, config_folder=None):
        self.config_folder = config_folder
        self.folder_name = folder_name

        self._logger = None
        self._test_dir = None
        self._args = None
        self._args_raw = None
        self._data_provider = None
        self._data_provider_test = None
        self._config_name = None
        self._loss_fn = None
        self._model = None

        self.mae_fn = torch.nn.L1Loss(reduction='mean')
        self.mse_fn = torch.nn.MSELoss(reduction='mean')

    def info(self, msg: str):
        self.logger.info(msg)

    @property
    def model(self):
        if self._model is None:
            use_swag = (self.args["uncertainty_modify"].split('_')[0] == 'swag')
            if use_swag:
                net = PhysDimeNet(**self.args)
                net = net.to(get_device())
                net = net.type(floating_type)
                net = SWAG(net, no_cov_mat=False, max_num_models=20)
                model_data = torch.load(os.path.join(self.folder_name, 'swag_model.pt'), map_location=get_device())
                net.load_state_dict(model_data)
            else:
                model_data = torch.load(os.path.join(self.folder_name, 'best_model.pt'), map_location=get_device())
                # temp fix, to be removed
                # model_data = fix_model_keys(model_data)
                net = init_model_test(self.args, model_data)
            self._model = net
        return self._model

    @property
    def loss_fn(self):
        if self._loss_fn is None:
            w_e, w_f, w_q, w_p = 1, self.args["force_weight"], self.args["charge_weight"], self.args["dipole_weight"]
            _loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p, action=self.args["action"],
                              auto_sol=self.args["auto_sol"], target_names=self.args["target_names"],
                              config_dict=self.args)
            self._loss_fn = _loss_fn
        return self._loss_fn

    @property
    def args(self):
        if self._args is None:
            _args = copy.deepcopy(self.args_raw)

            if _args["ext_atom_features"] is not None:
                # infer the dimension of external atom feature
                ext_atom_feature = getattr(self.ds[[0]].data, _args["ext_atom_features"])
                ext_atom_dim = ext_atom_feature.shape[-1]
                _args["ext_atom_dim"] = ext_atom_dim
                del ext_atom_feature

            inferred_prefix = self.folder_name.split('_run_')[0]
            if _args["folder_prefix"] != inferred_prefix:
                # print('overwriting folder {} ----> {}'.format(_args["folder_prefix"], inferred_prefix))
                _args["folder_prefix"] = inferred_prefix
            _args["requires_atom_prop"] = True

            self._args = _args
        return self._args

    @property
    def ds(self):
        if self._data_provider is None:
            _data_provider = ds_from_args(self.args_raw)

            # The lines below are dealing with the logic that I separate some test set from training set into
            # different files, which makes the code messy. It is not used in my relatively new datasets.
            if isinstance(_data_provider, tuple):
                _data_provider_test = _data_provider[1]
                _data_provider = _data_provider[0]
            else:
                _data_provider_test = _data_provider
            self._data_provider = _data_provider
            self._data_provider_test = _data_provider_test
        return self._data_provider

    @property
    def ds_test(self):
        if self._data_provider_test is None:
            # it was inited in self.data_provider
            __ = self.ds
        return self._data_provider_test

    @property
    def args_raw(self):
        if self._args_raw is None:
            if self.config_folder is not None:
                _args_raw, _config_name = read_folder_config(self.config_folder)
            else:
                _args_raw, _config_name = read_folder_config(self.folder_name)
            self._args_raw = _args_raw
            self._config_name = _config_name
        return self._args_raw

    @property
    def config_name(self):
        if self._config_name is None:
            __ = self.args_raw
        return self._config_name

    @property
    def save_root(self):
        raise NotImplementedError

    @property
    def logger(self):
        if self._logger is None:
            remove_handler()
            logging.basicConfig(filename=os.path.join(self.save_root, "test.log"),
                                format='%(asctime)s %(message)s', filemode='w')
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            self._logger = logger
        return self._logger


def ds_from_args(args, rm_keys=True):
    default_kwargs = {'root': args["data_root"], 'pre_transform': my_pre_transform, 'record_long_range': True,
                      'type_3_body': 'B', 'cal_3body_term': True}
    dataset_cls, _kwargs = data_provider_solver(args["data_provider"], default_kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    dataset = dataset_cls(**_kwargs)
    if rm_keys:
        dataset = remove_extra_keys(dataset)
    print("used dataset: {}".format(dataset.processed_file_names))
    if dataset.train_index is not None:
        validate_index(dataset.train_index, dataset.val_index, dataset.test_index)
    if ("add_sol" not in _kwargs or not _kwargs["add_sol"]) and args["data_provider"].split('[')[0] in ["frag9to20_all",
                                                                                                        "frag20_eMol9_combine"]:
        # for some weird logic, I separated training and testing dataset for those two datasets, so I have to deal with
        # it.
        logging.info("swapping {} to frag9to20_jianing".format(args["data_provider"]))
        frag20dataset, _kwargs = data_provider_solver('frag9to20_jianing', _kwargs)
        _kwargs["training_option"] = "test"
        print(_kwargs)
        return dataset, frag20dataset(**_kwargs)
    else:
        return dataset


def read_folder_config(folder_name):
    # parse config file
    if osp.exists(osp.join(folder_name, "config-test.txt")):
        config_name = osp.join(folder_name, "config-test.txt")
    else:
        config_name = glob.glob(osp.join(folder_name, 'config-*.txt'))[0]
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    args, unknown = parser.parse_known_args(["@" + config_name])
    args = vars(args)
    args = preprocess_config(args)
    return args, config_name
