import logging
import logging
import math
import os
import os.path as osp
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from DataPrepareUtils import rm_atom
from deprecated_code.junk_code import val_step
from train import val_step_new
from utils.trained_folder import TrainedFolder
from utils.utils_functions import remove_handler, collate_fn


class Tester(TrainedFolder):
    """
    Load a trained folder for performance evaluation
    """

    def __init__(self, folder_name, n_forward=5, x_forward=False, use_exist=False, check_active=False,
                 ignore_train=True, lightweight=True, config_folder=None, no_runtime_split=False):
        super().__init__(folder_name, config_folder)
        self.lightweight = lightweight
        self.ignore_train = ignore_train
        self.check_active = check_active
        self.no_runtime_split = no_runtime_split
        self.use_exist = use_exist
        self.x_forward = x_forward
        self.n_forward = n_forward

    @property
    def save_root(self) -> str:
        if self._test_dir is None:
            test_prefix = self.args["folder_prefix"] + '_test_'
            current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            tmp = test_prefix + current_time
            self._test_dir = osp.join(self.folder_name, osp.basename(tmp))
            os.mkdir(self._test_dir)
        return self._test_dir

    def run_test(self):
        # shutil.copyfile(os.path.join(folder_name, 'loss_data.pt'), os.path.join(test_dir, 'loss_data.pt'))
        shutil.copy(self.config_name, self.save_root)

        self.info("dataset in args: {}".format(self.args["data_provider"]))

        # ---------------------------------- Testing -------------------------------- #

        if self.ds.train_index is not None and \
                len(self.args["remove_atom_ids"]) > 0 and self.args["remove_atom_ids"][0] > 0:
            __, self.ds.val_index, _ = rm_atom(self.args["remove_atom_ids"], self.ds, ("valid",),
                                               (None, self.ds.val_index, None))
            __, __, self.ds.test_index = rm_atom(self.args["remove_atom_ids"], self.ds_test, ('test',),
                                                 (None, None, self.ds.test_index))

        self.info("dataset: {}".format(self.ds.processed_file_names))
        self.info("dataset test: {}".format(self.ds_test.processed_file_names))

        if not self.no_runtime_split and osp.exists(osp.join(self.folder_name, "runtime_split.pt")):
            explicit_split = torch.load(osp.join(self.folder_name, "runtime_split.pt"))
            # now I suffer the result from the "crime" of mixing "valid" with "val"
            explicit_split["val_index"] = explicit_split["valid_index"]
        else:
            explicit_split = None

        for index_name in ["train_index", "val_index", "test_index"]:
            self.info(f"Testing on {index_name}")
            if self.ignore_train and index_name == "train_index":
                continue
            if index_name == "test_index":
                this_ds = self.ds_test
            else:
                this_ds = self.ds
            index_name_ = index_name.split("_")[0]

            if explicit_split is not None:
                this_index = explicit_split[index_name]
            else:
                this_index = getattr(this_ds, index_name)
            if this_index is None:
                # for external test datasets where train_index and val_index are None
                continue
            this_index = torch.as_tensor(this_index)

            self.info("{} size: {}".format(index_name, len(this_index)))

            this_dl = DataLoader(this_ds[torch.as_tensor(this_index)], batch_size=self.args["valid_batch_size"],
                                 collate_fn=collate_fn, pin_memory=False, shuffle=False)
            this_info, this_std = test_step(self.args, self.model, this_dl, len(this_index),
                                            loss_fn=self.loss_fn, mae_fn=self.mae_fn, mse_fn=self.mse_fn,
                                            dataset_name='{}_{}'.format(self.args["data_provider"], index_name_),
                                            run_dir=self.save_root, n_forward=self.n_forward, action=self.args["action"],
                                            lightweight=self.lightweight)
            self.info("-------------- {} ---------------".format(index_name_))
            for key in this_info:
                self.info("{}: {}".format(key, this_info[key]))
            self.info("----------- end of {} ------------".format(index_name_))

        # remove global variables
        remove_handler()


def print_uncertainty_figs(pred_std, diff, name, unit, test_dir, n_bins=10):
    import matplotlib.pyplot as plt
    # let x axis ordered ascending
    std_rank = torch.argsort(pred_std)
    pred_std = pred_std[std_rank]
    diff = diff[std_rank]

    diff = diff.abs()
    diff_2 = diff ** 2
    x_data = torch.arange(pred_std.min(), pred_std.max(), (pred_std.max() - pred_std.min()) / n_bins)
    mae_data = torch.zeros(x_data.shape[0] - 1).float()
    rmse_data = torch.zeros_like(mae_data)
    for i in range(x_data.shape[0] - 1):
        mask = (pred_std < x_data[i + 1]) & (pred_std > x_data[i])
        mae_data[i] = diff[mask].mean()
        rmse_data[i] = torch.sqrt(diff_2[mask].mean())

    plt.figure(figsize=(15, 10))
    # Plotting predicted error MAE vs uncertainty
    plt.plot(x_data[1:], mae_data, label='{} MAE, {}'.format(name, unit))
    plt.plot(x_data[1:], rmse_data, label='{} RMSE, {}'.format(name, unit))
    plt.legend()
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('Error of {}, {}'.format(name, unit))
    plt.savefig(os.path.join(test_dir, 'avg_error_uncertainty'))

    fig, ax1 = plt.subplots(figsize=(15, 10))
    diff_abs = diff.abs()
    ax1.scatter(pred_std, diff_abs, alpha=0.1)
    ax1.set_xlabel('Uncertainty of {}, {}'.format(name, unit))
    ax1.set_ylabel('Error of {}, {}'.format(name, unit))
    # plt.title('Uncertainty vs. prediction error')
    # plt.savefig(os.path.join(test_dir, 'uncertainty'))

    # Plotting cumulative large error percent vs uncertainty
    thresholds = ['0', '1.0', '10.0']
    cum_large_count = {threshold: torch.zeros_like(x_data) for threshold in thresholds}
    for i in range(x_data.shape[0]):
        mask = (pred_std < x_data[i])
        select_diff = diff[mask]
        for threshold in thresholds:
            cum_large_count[threshold][i] = select_diff[select_diff > float(threshold)].shape[0]
    # plt.figure(figsize=(15, 10))
    ax2 = ax1.twinx()
    for threshold in thresholds:
        count = cum_large_count[threshold]
        ax2.plot(x_data, count / count[-1] * 100,
                 label='%all molecules' if threshold == '0' else '%large>{}kcal/mol'.format(threshold))

    plt.legend()
    # ax2.xlabel('Uncertainty of {}, {}'.format(name, unit))
    ax2.set_ylabel('percent of molecules')
    plt.savefig(os.path.join(test_dir, 'percent'))

    # Plotting density of large error percent vs uncertainty
    x_mid = (x_data[:-1] + x_data[1:]) / 2
    plt.figure(figsize=(15, 10))
    for i, threshold in enumerate(thresholds):
        count_density_all = cum_large_count['0'][1:] - cum_large_count['0'][:-1]
        count_density = cum_large_count[threshold][1:] - cum_large_count[threshold][:-1]
        count_density_lower = count_density_all - count_density
        width = (x_data[1] - x_data[0]) / (len(thresholds) * 5)
        plt.bar(x_mid + i * width, count_density / count_density.sum() * 100, width=width,
                label='all molecules' if threshold == '0' else 'large>{}kcal/mol'.format(threshold))
        # if threshold != '0':
        #     plt.bar(x_mid - i * width, count_density_lower / count_density_lower.sum() * 100, width=width,
        #             label='large<{}kcal/mol'.format(threshold))
    plt.legend()
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('density of molecules')
    plt.xticks(x_data)
    plt.savefig(os.path.join(test_dir, 'percent_density'))

    # box plot
    num_points = diff.shape[0]
    step = math.ceil(num_points / n_bins)
    sep_index = torch.arange(0, num_points + 1, step)
    y_blocks = []
    x_mean = []
    for num, start in enumerate(sep_index):
        if num + 1 < sep_index.shape[0]:
            end = sep_index[num + 1]
        else:
            end = num_points
        y_blocks.append(diff[start: end].numpy())
        x_mean.append(pred_std[start: end].mean().item())
    plt.figure(figsize=(15, 10))
    box_size = (0.30 * (x_mean[-1] - x_mean[0]) / len(x_mean))
    plt.boxplot(y_blocks, notch=True, positions=x_mean, vert=True, showfliers=False,
                widths=box_size)
    plt.xticks(x_mean, ['{:.3f}'.format(_x_mean) for _x_mean in x_mean])
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlim([x_mean[0] - box_size, x_mean[-1] + box_size])
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('Error of {}, {}'.format(name, unit))
    plt.title('Boxplot')
    plt.savefig(os.path.join(test_dir, 'box_plot'))

    # interval percent
    small_err_percent_1 = np.asarray([(x < 1.).sum() / x.shape[0] for x in y_blocks])
    large_err_percent_10 = np.asarray([(x > 10.).sum() / x.shape[0] for x in y_blocks])
    x_mean = np.asarray(x_mean)
    plt.figure(figsize=(15, 10))
    bar_size = (0.30 * (x_mean[-1] - x_mean[0]) / len(x_mean))
    plt.bar(x_mean - bar_size, large_err_percent_10, label='percent, error > 10kcal/mol', width=bar_size)
    plt.bar(x_mean, 1 - large_err_percent_10 - small_err_percent_1,
            label='percent, 1kcal/mol < error < 10kcal/mol', width=bar_size)
    plt.bar(x_mean + bar_size, small_err_percent_1, label='small error < 1kcal/mol', width=bar_size)
    plt.legend()
    plt.xticks(x_mean, ['{:.3f}'.format(_x_mean) for _x_mean in x_mean])
    plt.xlim([x_mean[0] - box_size, x_mean[-1] + box_size])
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('percent')
    plt.title('error percent')
    plt.savefig(os.path.join(test_dir, 'error_percent'))
    return


def test_info_analyze(pred, target, test_dir, logger=None, name='Energy', threshold_base=1.0, unit='kcal/mol',
                      pred_std=None, x_forward=0):
    diff = pred - target
    rank = torch.argsort(diff.abs())
    diff_ranked = diff[rank]
    if logger is None:
        logging.basicConfig(filename=os.path.join(test_dir, 'test.log'),
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        remove_logger = True
    else:
        remove_logger = False
    logger.info('Top 10 {} error: {}'.format(name, diff_ranked[-10:]))
    logger.info('Top 10 {} error id: {}'.format(name, rank[-10:]))
    e_mae = diff.abs().mean()
    logger.info('{} MAE: {}'.format(name, e_mae))
    thresholds = torch.logspace(-2, 2, 50) * threshold_base
    thresholds = thresholds.tolist()
    thresholds.extend([1.0, 10.])
    for threshold in thresholds:
        mask = (diff.abs() < threshold)
        logger.info('Percent of {} error < {:.4f} {}: {} out of {}, {:.2f}%'.format(
            name, threshold, unit, len(diff[mask]), len(diff), 100 * float(len(diff[mask])) / len(diff)))
    torch.save(diff, os.path.join(test_dir, 'diff.pt'))

    # concrete dropout
    if x_forward and (pred_std is not None):
        # print_scatter(pred_std, diff, name, unit, test_dir)
        print_uncertainty_figs(pred_std, diff, name, unit, test_dir)
    if x_forward:
        raise NotImplemented
        # print_training_curve(test_dir)

    if remove_logger:
        remove_handler(logger)
    return


def test_step(args, net, data_loader, total_size, loss_fn, mae_fn=torch.nn.L1Loss(reduction='mean'),
              mse_fn=torch.nn.MSELoss(reduction='mean'), dataset_name='data', run_dir=None, lightweight=True,
              n_forward=50, **kwargs):
    if args["uncertainty_modify"] == 'none':
        result = val_step_new(net, data_loader, loss_fn, diff=True, lightweight=lightweight)
        # we don't want to create folders here
        dataset_name = dataset_name.replace('\\', '.')
        dataset_name = dataset_name.replace('/', '.')
        torch.save(result, os.path.join(run_dir, 'loss_{}.pt'.format(dataset_name)))
        return result, None
    elif args["uncertainty_modify"].split('_')[0].split('[')[0] in ['concreteDropoutModule', 'concreteDropoutOutput',
                                                                    'swag']:
        print("You need to update the code of val_step_new")
        if os.path.exists(os.path.join(run_dir, dataset_name + '-avg{}.pt'.format(n_forward))):
            print('loading exist files!')
            avg_result = torch.load(os.path.join(run_dir, dataset_name + '-avg{}.pt'.format(n_forward)))
            std_result = torch.load(os.path.join(run_dir, dataset_name + '-std{}.pt'.format(n_forward)))
        else:
            avg_result = {}
            std_result = {}
            cum_result = {'E_pred': [], 'D_pred': [], 'Q_pred': []}
            for i in range(n_forward):
                if args["uncertainty_modify"].split('_')[0] == 'swag':
                    net.sample(scale=1.0, cov=True)
                result_i = val_step(net, data_loader, total_size, loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn,
                                    dataset_name=dataset_name, print_to_log=False, detailed_info=True, **kwargs)

                for key in cum_result.keys():
                    cum_result[key].append(result_i[key])
            # list -> tensor
            for key in cum_result.keys():
                cum_result[key] = torch.stack(cum_result[key])
                avg_result[key] = cum_result[key].mean(dim=0)
                std_result[key] = cum_result[key].std(dim=0)
            torch.save(avg_result, os.path.join(run_dir, dataset_name + '-avg{}.pt'.format(n_forward)))
            torch.save(std_result, os.path.join(run_dir, dataset_name + '-std{}.pt'.format(n_forward)))
        return avg_result, std_result
    else:
        raise ValueError('unrecognized uncertainty_modify: {}'.format(args.uncertainty_modify))
