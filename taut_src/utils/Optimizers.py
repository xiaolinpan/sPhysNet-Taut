import time

import torch
import copy

from torch.nn import Module
from torch.optim.swa_utils import AveragedModel

# class _RequiredParameter(object):
#     """Singleton class representing a required parameter for an Optimizer."""
#
#     def __repr__(self):
#         return "<required parameter>"
#
#
# required = _RequiredParameter()
from utils.time_meta import record_data
from utils.utils_functions import get_device


class EmaAmsGrad(torch.optim.Adam):
    def __init__(self, training_model: torch.nn.Module, lr=1e-3, betas=(0.9, 0.99),
                 eps=1e-8, weight_decay=0, ema=0.999, shadow_dict=None):
        super().__init__(training_model.parameters(), lr, betas, eps, weight_decay, amsgrad=True)
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


class MySGD(torch.optim.SGD):
    """
    my wrap of SGD for compatibility issues
    """

    def __init__(self, model, *args, **kwargs):
        self.shadow_model = model
        super(MySGD, self).__init__(model.parameters(), *args, **kwargs)

    def step(self, closure=None):
        return super(MySGD, self).step(closure)
