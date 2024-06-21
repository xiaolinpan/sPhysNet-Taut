class Tag:
    """
    Put tags all together for easier management.
    """
    def __init__(self):
        pass

    @property
    def requires_atomic_prop(self):
        return ["names_atomic"]

    @property
    def step_per_step(self):
        return ["StepLR"]

    @property
    def step_per_epoch(self):
        return ["ReduceLROnPlateau"]

    @property
    def loss_metrics(self):
        return ["mae", "rmse", "mse", "ce", "evidential"]

    @staticmethod
    # in validation step: concat result
    def val_concat(key):
        return key.startswith("DIFF") or key in ["RAW_PRED", "LABEL", "atom_embedding", "ATOM_MOL_BATCH", "ATOM_Z",
                                                 "PROP_PRED", "PROP_TGT", "UNCERTAINTY", "Z_PRED"]

    @staticmethod
    # in validation step: calculate average
    def val_avg(key):
        return key.startswith("MAE") or key.startswith("MSE") or key in ["accuracy", "z_loss"]


tags = Tag()
