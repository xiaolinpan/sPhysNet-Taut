import argparse
import os
import os.path as osp
import time

import pandas as pd
import torch
from ase.units import Hartree, eV
from torch_geometric.data import Data

from utils import gauss
import utils.gauss.read_gauss_log
from utils.DataPrepareUtils import my_pre_transform
from utils.sdf2pyg_single import sdf2pyg_single

hartree2ev = Hartree / eV
# relative to the "sPhysNet-MT" directory since the code will be executed there
# you can also use absolute path if you are confused
RAW_DATA_ROOT = "./data/raw/Frag20_sdfs"
# TODO: remove this line after finishing debugging
if osp.exists("/vast/sx801/sPhysNet-MT-data/raw"):
    RAW_DATA_ROOT = "/vast/sx801/sPhysNet-MT-data/raw"
# temp data for data preprocessing
TEMP_DATA_ROOT = osp.join(RAW_DATA_ROOT, "..", "tmp")
PROCESSED_DATA_ROOT = osp.join(RAW_DATA_ROOT, "..", "processed")
os.makedirs(TEMP_DATA_ROOT, exist_ok=True)
os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)

def get_debug_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def frag20_csd20_sol_single(sample_id, csv, geometry, out_path, df=None):
    # t0 = time.time()
    if df is None:
        df = pd.read_csv(csv)
    this_sdf = sdf_from_sample_id(sample_id, geometry)

    # print(f"init: {time.time()-t0}")
    sdf2pyg_single(this_sdf, df, out_path)


def sdf_from_sample_id(sample_id, geometry):
    source = sample_id.split("_")[0]
    idx = sample_id.split("_")[1]

    if geometry == "qm":
        extra = ".opt"
    else:
        extra = ""

    try:
        source = int(source)
        this_sdf = f"{RAW_DATA_ROOT}/Frag20_sdfs/Frag20_{source}_data/{idx}{extra}.sdf"
    except ValueError:
        assert source in ["PubChem", "Zinc", "CCDC"]
        if source in ["PubChem", "Zinc"]:
            this_sdf = f"{RAW_DATA_ROOT}/Frag20_sdfs/Frag20_9_data/{source.lower()}/{idx}{extra}.sdf"
        else:
            if geometry == "qm":
                extra = ".opt"
            else:
                extra = "_min"
            this_sdf = f"{RAW_DATA_ROOT}/CSD20_sdfs/cry_min/{idx}{extra}.sdf"
    return this_sdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="20_11110")
    parser.add_argument("--csv", default="20_11110.csv")
    parser.add_argument("--geometry", default="qm")
    parser.add_argument("--out_path", default="20_11110.pyg")
    args = parser.parse_args()
    args = vars(args)

    frag20_csd20_sol_single(**args)


if __name__ == '__main__':
    main()

