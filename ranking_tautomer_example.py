#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
import pandas as pd
from predict_tautomer import get_taut_data
import mols2grid

def construct_data(out_data):
    new_data = []
    for info in out_data:
        if info["label"] == "high_energy":
            continue
        new_data.append([Chem.MolFromSmiles(info["tsmi"]), info["tsmi"], info["score"], info["label"]])
    df_data = pd.DataFrame(new_data)
    df_data.columns = ["tmol", "tsmi", "score", "label"]
    return df_data


import glob
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

def uncharge_molecule(mol):
    un = rdMolStandardize.Uncharger()
    mol = un.uncharge(mol)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return mol

ligand_files = glob.glob("../scoringfunction/DataSet/Structure/*/Lig_fixed.sdf")

cutmol = True
num_confs = 50
energy_cutoff = 2.8
ph = 7.0
tph = 1.5


num = 0
for file in ligand_files[3000:]:
    mol = next(Chem.SDMolSupplier( file ))
    if not mol:
        print("read mol error:", file)
    try:
        mol = uncharge_molecule(mol)
    except:
        print("uncharge error mol:", file)
        continue
    smi = Chem.MolToSmiles( mol, isomericSmiles=False)
    try:
        data = get_taut_data(smi, cutmol, num_confs, energy_cutoff, ph, tph)
        df = construct_data(data)
    except:
        print("error mol:", file)
        continue
    if smi not in df['tsmi'].tolist():
        print(file, smi)
        num += 1
