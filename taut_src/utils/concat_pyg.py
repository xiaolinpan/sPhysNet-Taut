import argparse
import copy
import glob
import os
import os.path as osp
import time
import traceback
from tqdm import tqdm

import pandas as pd
import torch
import torch_geometric.data


def concat_pyg(pygs: list = None, save_pyg: str = None, data_list=None, save_split=None, extend_load=False,
               ref_csv=None, all_train=False, all_test=False, del_keys=None, **kwargs):
    train_index = []
    test_index = []

    if ref_csv is not None:
        ref_df = pd.read_csv(ref_csv, dtype={"sample_id": str, "FileHandle": str}) \
            .rename({"FileHandle": "sample_id"}, axis=1).set_index("sample_id")
    else:
        ref_df = None

    t0 = time.time()
    if data_list is None:
        data_list = []
        for pyg in tqdm(pygs, desc="loading data into memory"):
            try:
                d = torch.load(pyg)

                if ref_df is not None:
                    sample_id = str(osp.basename(pyg).split(".")[0])
                    this_info = ref_df.loc[[sample_id], :]
                    for key in this_info.columns:
                        if extend_load:
                            for _d in d:
                                setattr(_d, key, this_info[key].item())
                        else:
                            setattr(d, key, this_info[key].item())
                
                if del_keys is not None:
                    for key in del_keys:
                        delattr(d, key)

                if extend_load:
                    data_list.extend(d)
                else:
                    data_list.append(d)
            except Exception as e:
                print(f"Error loading {pyg}: {e}")
                print(traceback.format_exc())

    print(f"Total time loading data into memory: {time.time() - t0}]")
    t0 = time.time()

    os.makedirs(osp.dirname(save_pyg), exist_ok=True)
    if save_split is not None:
        if all_train:
            split = {
                "train_index": torch.arange(len(data_list)),
                "val_index": None,
                "test_index": None
            }
        elif all_test:
            split = {
                "train_index": None,
                "val_index": None,
                "test_index": torch.arange(len(data_list))
            }
        else:
            for num, pyg in enumerate(data_list):
                if pyg.split == "train":
                    train_index.append(num)
                else:
                    test_index.append(num)
            split = {
                "train_index": torch.as_tensor(train_index),
                "val_index": None,
                "test_index": torch.as_tensor(test_index)
            }
        torch.save(split, save_split)

    print(f"Total time saving split: {time.time() - t0}]")
    t0 = time.time()

    data_concat = torch_geometric.data.InMemoryDataset.collate(data_list)
    print(data_concat)
    torch.save(data_concat, save_pyg)

    print(f"Total time collate: {time.time() - t0}]")
    t0 = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pygs", default=None)
    parser.add_argument("--pyg_folder", default=None)
    parser.add_argument("--save_pyg")
    parser.add_argument("--save_split", default=None)
    parser.add_argument("--extend_load", action="store_true")
    parser.add_argument("--ref_csv", default=None)
    parser.add_argument("--all_train", action="store_true")
    parser.add_argument("--all_test", action="store_true")

    args = parser.parse_args()
    args = vars(args)
    processed_args = copy.deepcopy(args)

    if args["pygs"] is not None:
        num = None
        for name in ["pygs"]:
            if args[name] is None:
                processed_args[name] = [None] * num
                continue

            with open(args[name]) as f:
                processed_args[name] = f.read().split()

            if num is None:
                num = len(processed_args[name])

    if args["pyg_folder"] is not None:
        pygs = glob.glob(osp.join(args["pyg_folder"], "*.pyg"))
        pygs.sort()
        processed_args["pygs"] = pygs

    concat_pyg(**processed_args)


if __name__ == '__main__':
    main()
