from copy import deepcopy
from typing import Union, List

import torch
import torch_scatter

from utils.tags import tags
from utils.utils_functions import kcal2ev, evidential_loss_new
from sklearn.metrics import roc_auc_score

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15


class LossFn:
    def __init__(self, w_e, w_f, w_q, w_p, action: Union[List[str], str] = "E", auto_sol=False, target_names=None,
                 config_dict=None):
        self.loss_metric = config_dict["loss_metric"].lower()
        self.z_loss_weight = config_dict["z_loss_weight"]
        # keep only solvation energy/logP, only used in transfer learning on exp datasets
        self.keep = config_dict["keep"]
        self.mask_atom = config_dict["mask_atom"]
        self.flex_sol = config_dict["flex_sol"]
        self.loss_metric_upper = self.loss_metric.upper()
        assert self.loss_metric in tags.loss_metrics

        self.target_names = deepcopy(target_names)
        self.action = deepcopy(action)
        self.w_e = w_e
        self.w_f = w_f
        self.w_q = w_q
        self.w_d = w_p
        self.auto_sol = auto_sol
        if self.auto_sol:
            if "watEnergy" in self.target_names and "gasEnergy" in self.target_names:
                self.target_names.append("CalcSol")
            if "octEnergy" in self.target_names and "gasEnergy" in self.target_names:
                self.target_names.append("CalcOct")
            if "watEnergy" in self.target_names and "octEnergy" in self.target_names:
                self.target_names.append("watOct")
        self._target_name_to_id = None

        if self.loss_metric == "ce":
            assert self.action in ["names", "names_atomic"]
            assert len(self.target_names) == 1, "Currently only support single task classification"
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.loss_metric == "evidential":
            self.evi_loss = evidential_loss_new
            self.soft_plus = torch.nn.Softplus()
        self.num_targets = len(self.target_names)

    def __call__(self, model_output, data, is_training, loss_detail=False, diff_detail=False):
        if self.loss_metric == "ce":
            prop_tgt, prop_pred = self._get_target(model_output, data)
            assert prop_tgt.shape[-1] == 1, "Currently only support single task classification"
            prop_tgt = prop_tgt.view(-1)
            total_loss = self.ce_loss(prop_pred, prop_tgt)
            if not loss_detail:
                return total_loss
            else:
                label_pred = torch.argmax(prop_pred, dim=-1)
                accuracy = (label_pred == prop_tgt).sum() / len(label_pred)

                # n_units: number of molecules
                detail = {"n_units": data.N.shape[0], "accuracy": float(accuracy)}
                if diff_detail:
                    detail["RAW_PRED"] = prop_pred.detach().cpu()
                    detail["LABEL"] = prop_tgt.cpu()
                    detail["ATOM_MOL_BATCH"] = data.atom_mol_batch.detach().cpu()
                return total_loss, detail

        if self.action in ["names", "names_and_QD", "names_atomic"]:
            detail = {}

            prop_tgt, prop_pred = self._get_target(model_output, data)

            evi_cal_dict = {}
            if self.loss_metric == "evidential":
                min_val = 1e-6
                means, log_lambdas, log_alphas, log_betas = torch.split(prop_pred, prop_pred.shape[-1] // 4, dim=-1)
                lambdas = self.soft_plus(log_lambdas) + min_val
                # add 1 for numerical contraints of Gamma function
                alphas = self.soft_plus(log_alphas) + min_val + 1
                betas = self.soft_plus(log_betas) + min_val
                evi_cal_dict["mu"] = means
                evi_cal_dict["v"] = lambdas
                evi_cal_dict["alpha"] = alphas
                evi_cal_dict["beta"] = betas
                prop_pred = means
                evi_cal_dict["targets"] = prop_tgt

            if self.flex_sol:
                mask = data.mask
                batch = torch.ones_like(mask).long()
                batch[mask] = 0
                diff = prop_pred - prop_tgt
                mae_loss = torch_scatter.scatter_mean(diff.abs(), batch, dim=0)[[0], :]
                mse_loss = torch_scatter.scatter_mean(diff ** 2, batch, dim=0)[[0], :]
                rmse_loss = torch.sqrt(mse_loss)

                flex_units = {}
                for i, key in enumerate(self.target_names):
                    flex_units[key] = (mask[:, i]).sum().item()
                detail["flex_units"] = flex_units
            else:
                mae_loss = torch.mean(torch.abs(prop_pred - prop_tgt), dim=0, keepdim=True)
                mse_loss = torch.mean((prop_pred - prop_tgt) ** 2, dim=0, keepdim=True)
                rmse_loss = torch.sqrt(mse_loss)

            if self.loss_metric == "mae":
                total_loss = mae_loss.sum()
            elif self.loss_metric == "mse":
                total_loss = mse_loss.sum()
            elif self.loss_metric == "rmse":
                total_loss = rmse_loss.sum()
            elif self.loss_metric == "evidential":
                total_loss = self.evi_loss(**evi_cal_dict).sum()
            else:
                raise ValueError("Invalid total loss: " + self.loss_metric)

            if loss_detail:
                # record details including MAE, RMSE, Difference, etc..
                # It is required while valid and test step but not required in training
                for i, name in enumerate(self.target_names):
                    detail["MAE_{}".format(name)] = mae_loss[:, i].item()
                    detail["MSE_{}".format(name)] = mse_loss[:, i].item()
                if diff_detail:
                    detail["PROP_PRED"] = prop_pred.detach().cpu()
                    detail["PROP_TGT"] = prop_tgt.detach().cpu()
                    if self.loss_metric == "evidential":
                        detail["UNCERTAINTY"] = (betas / (lambdas * (alphas-1))).detach().cpu()
            else:
                detail = None

            if self.action == "names_and_QD":
                if self.loss_metric == "mae":
                    q_loss = torch.mean(torch.abs(model_output["Q_pred"] - data.Q))
                    d_loss = torch.mean(torch.abs(model_output["D_pred"] - data.D))
                else:
                    q_loss = torch.mean((model_output["Q_pred"] - data.Q) ** 2)
                    d_loss = torch.mean((model_output["D_pred"] - data.D) ** 2)
                    if self.loss_metric == "rmse":
                        q_loss = torch.sqrt(q_loss)
                        d_loss = torch.sqrt(d_loss)
                total_loss = total_loss + self.w_q * q_loss + self.w_d * d_loss
                if loss_detail:
                    detail["{}_Q".format(self.loss_metric_upper)] = q_loss.item()
                    detail["{}_D".format(self.loss_metric_upper)] = d_loss.item()
                    if diff_detail:
                        detail["DIFF_Q"] = (model_output["Q_pred"] - data.Q).detach().cpu().view(-1)
                        detail["DIFF_D"] = (model_output["D_pred"] - data.D).detach().cpu().view(-1)

            if self.z_loss_weight > 0:
                assert "first_layer_vi" in model_output
                z_loss = self.ce_loss(model_output["first_layer_vi"], data.Z)
                total_loss = total_loss + self.z_loss_weight * z_loss
                if loss_detail:
                    detail["z_loss"] = z_loss.item()
                    detail["Z_PRED"] = torch.argmax(model_output["first_layer_vi"].detach().cpu(), dim=-1)

            if loss_detail:
                # n_units: number of molecules
                detail["n_units"] = data.N.shape[0]
                detail["ATOM_MOL_BATCH"] = data.atom_mol_batch.detach().cpu()
                detail["ATOM_Z"] = data.Z.detach().cpu()
                for key in ["atom_embedding"]:
                    if key in model_output.keys():
                        detail[key] = model_output[key].detach().cpu()
                return total_loss, detail
            else:
                return total_loss

        elif self.action == "E":
            # default PhysNet setting
            assert self.loss_metric == "mae"
            E_loss, F_loss, Q_loss, D_loss = 0, 0, 0, 0
            E_loss = self.w_e * torch.mean(torch.abs(model_output["mol_prop"] - data.E))

            # if 'F' in data.keys():
            #     F_loss_batch = torch_geometric.utils.scatter_('mean', torch.abs(F_pred - data['F'].to(device)),
            #                                                   data['atom_to_mol_batch'].to(device))
            #     F_loss = self.w_f * torch.sum(F_loss_batch) / 3

            Q_loss = self.w_q * torch.mean(torch.abs(model_output["Q_pred"] - data.Q))

            D_loss = self.w_d * torch.mean(torch.abs(model_output["D_pred"] - data.D))

            if loss_detail:
                return E_loss + F_loss + Q_loss + D_loss, {"MAE_E": E_loss.item(), "MAE_F": F_loss,
                                                           "MAE_Q": Q_loss.item(), "MAE_D": D_loss.item(),
                                                           "DIFF_E": (model_output[
                                                                          "mol_prop"] - data.E).detach().cpu().view(-1)}
            else:
                return E_loss + F_loss + Q_loss + D_loss
        else:
            raise ValueError("Invalid action: {}".format(self.action))

    def _get_target(self, model_output: dict, data):
        """
        Get energy target from data
        Solvation energy is in kcal/mol but gas/water/octanol energy is in eV
        """
        # multi-task prediction
        if self.action in tags.requires_atomic_prop:
            # TODO predict atom and mol prop at the same time
            prop_name = "atom_prop"
        else:
            prop_name = "mol_prop"
        prop_pred = model_output[prop_name]
        if "mol_prop_pool" in model_output.keys():
            prop_pred = torch.cat([prop_pred, model_output["mol_prop_pool"]], dim=-1)
        if self.auto_sol:
            total_pred = [prop_pred]
            if "gasEnergy" in self.target_name_to_id.keys():
                target_name_to_id = self.target_name_to_id
            else:
                assert self.keep
                target_name_to_id = {
                    "gasEnergy": 0, "watEnergy": 1, "octEnergy": 2
                }
            for sol_name in ["watEnergy", "octEnergy"]:
                if sol_name in target_name_to_id.keys():
                    gas_id = target_name_to_id["gasEnergy"]
                    sol_id = target_name_to_id[sol_name]
                    # converting it to kcal/mol
                    total_pred.append((prop_pred[:, sol_id] - prop_pred[:, gas_id]).view(-1, 1) / kcal2ev)
            if "watEnergy" in target_name_to_id.keys() and "octEnergy" in target_name_to_id.keys():
                wat_id = target_name_to_id["watEnergy"]
                oct_id = target_name_to_id["octEnergy"]
                total_pred.append((prop_pred[:, wat_id] - prop_pred[:, oct_id]).view(-1, 1) / kcal2ev)
            prop_pred = torch.cat(total_pred, dim=-1)

        if not self.flex_sol:
            if self.keep == "waterSol":
                prop_pred = prop_pred[:, [3]]
            elif self.keep == "logP":
                prop_pred = prop_pred[:, [5]] / logP_to_watOct
            elif self.keep == "watOct":
                prop_pred = prop_pred[:, [5]]
            else:
                assert self.keep is None, f"Invalid keep arg: {self.keep}"

        prop_tgt = torch.cat([getattr(data, name).view(-1, 1) for name in self.target_names], dim=-1)
        if self.loss_metric not in ["ce", "evidential"]:
            assert prop_pred.shape[-1] == self.num_targets
        elif self.loss_metric == "evidential":
            # mu, v, alpha, beta
            assert prop_pred.shape[-1] == self.num_targets * 4

        if self.mask_atom:
            mask = data.mask.bool()
            prop_tgt = prop_tgt[mask, :]
            prop_pred = prop_pred[mask, :]
        return prop_tgt, prop_pred

    @property
    def target_name_to_id(self):
        if self._target_name_to_id is None:
            temp = {}
            for name in self.target_names:
                temp[name] = self.target_names.index(name)
            self._target_name_to_id = temp
        return self._target_name_to_id
