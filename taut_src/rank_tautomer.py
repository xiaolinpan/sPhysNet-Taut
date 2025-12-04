from typing import Any
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from multiprocessing import Pool
import torch
from taut_src.models import load_models
from taut_src.config import model_paths
from taut_src.calc_input import calc_data_for_predict
#from models import load_models
#from config import model_paths
#from calc_input import calc_data_for_predict
import warnings
warnings.filterwarnings("ignore")

models  = load_models(model_paths)


def get_device() -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def linker_to_carbon(smi: Any) -> Any:
    mol = Chem.MolFromSmiles(smi)

    linker_aids = []
    for at in mol.GetAtoms():
        if at.GetSymbol() == "*":
            idx = at.GetIdx()
            linker_aids.append(idx)

    emol = Chem.RWMol(mol)
    for idx in linker_aids:
        emol.ReplaceAtom(idx, Chem.Atom(6))
    nmol = emol.GetMol()
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(nmol)))
    return smi


def predict_ensemble(models: Any, data1: Any, data2: Any) -> Any:
    data1 = data1.to( get_device() )
    data2 = data2.to( get_device() )
    preds = []
    for net in models:
        output = net(data1, data2).item()
        preds.append( output )
    return np.mean(preds)


def predict_by_smis(smis: Any, num_confs: Any) -> Any:
    datas = []
    for idx, smi in enumerate(smis):
        data = calc_data_for_predict(smi, num_confs=num_confs)
        datas.append(data)
    
    output = []
    for idx in range(len(smis)):
        smi = smis[idx]
        if idx == 0:
            dG = 0.0
        else:
            dG = predict_ensemble(models, datas[0], datas[idx])
        output.append([idx, smi, dG])
    return output


def calc_scores(tauts: Any, num_confs: Any, is_fragment: Any) -> Any:
    if is_fragment:
        tauts_smis_include_linker = [Chem.MolToSmiles(taut.mol) for taut in tauts]
        tauts_smis_exclude_linker = [linker_to_carbon(smi) for smi in tauts_smis_include_linker]
        
        output = predict_by_smis(tauts_smis_exclude_linker, num_confs) 
        
        res = []
        for smi_idx, tsmi, dG in output:
            lsmi = tauts_smis_include_linker[smi_idx]
            res.append([lsmi, dG])
    else:
        tauts_smis = [taut.smi for taut in tauts]
        output = predict_by_smis(tauts_smis, num_confs)
        res = []
        for smi_idx, tsmi, dG in output:
            res.append([tsmi, dG])
    df = pd.DataFrame(res)
    if len(df) == 0:
        return df 
    df.columns = ["smi", "dG"] 
    return df


def rank_tauts(tauts: Any, num_confs: Any, is_fragment: Any=True) -> Any:
    df = calc_scores(tauts, num_confs, is_fragment)
    smirks_rules = [taut.smirks for taut in tauts]
    df["smirks"] = smirks_rules
    df = df.sort_values("dG")
    df["dG"] = df["dG"] - df["dG"].min()
    return df

        
if __name__=="__main__":
    #smi = "Brc1cnn2c1nc(cc2NCc1cccnc1)c1ccccc1"
    #smi = "Cc1n[nH]c(c12)OC(N)=C(C#N)C2(C(C)C)c(cc3C(F)(F)F)cc(c3)N4CCCC4"
    #smi = "CS(=O)(=O)c1ccc(cc1)c1cccn2c1nc(n2)Nc1ccc(cc1)N1CCOCC1"
    tauts = ["O=c1ccnc2o[nH]cc1-2", "O=c1cc[nH]c2oncc12"]
    df = predict_by_smis(tauts, num_confs=10) 
    #df = rank_tauts(tauts, num_confs=50, is_fragment=False)
    print(df)



