from utils.utils_functions import collate_fn


class CollateFunction:
    def __init__(self, config_dict):
        if config_dict["target_nodes"]:
            self.n_extra_nodes = config_dict["n_output"]
        else:
            self.n_extra_nodes = 0

    def __call__(self, data_list):
        return collate_fn(data_list, n_extra_node=self.n_extra_nodes)


