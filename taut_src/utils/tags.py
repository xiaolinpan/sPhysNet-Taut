from typing import Any
class Tag:
    """
    Put tags all together for easier management.
    """
    def __init__(self) -> None:
        pass

    @property
    def requires_atomic_prop(self) -> Any:
        return ["names_atomic"]

    @property
    def step_per_step(self) -> Any:
        return ["StepLR"]

    @property
    def step_per_epoch(self) -> Any:
        return ["ReduceLROnPlateau"]

    @property
    def loss_metrics(self) -> Any:
        return ["mae", "rmse", "mse", "ce", "evidential"]

    @staticmethod
    # in validation step: concat result
    def val_concat(key: Any) -> Any:
        return key.startswith("DIFF") or key in ["RAW_PRED", "LABEL", "atom_embedding", "ATOM_MOL_BATCH", "ATOM_Z",
                                                 "PROP_PRED", "PROP_TGT", "UNCERTAINTY", "Z_PRED"]

    @staticmethod
    # in validation step: calculate average
    def val_avg(key: Any) -> Any:
        return key.startswith("MAE") or key.startswith("MSE") or key in ["accuracy", "z_loss"]


tags = Tag()
