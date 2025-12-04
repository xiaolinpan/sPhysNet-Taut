from typing import Any
from rdkit import Chem
from collections import namedtuple

def get_linker_map_index(mol: Any, smi: Any, num: Any) -> Any:
    data = namedtuple("frag", "mol smi aidx_map_lidx")
    index_map = namedtuple("atom_idx_map_linker_idx", "atom_idx linker_idx")
    aidx_map_lidx, linker_idxs = [], []
    for at in mol.GetAtoms():
        if at.GetSymbol() == "*":
            aidx = at.GetNeighbors()[0].GetIdx()
            aidx_map_lidx.append(index_map(atom_idx=num+aidx, linker_idx=at.GetAtomMapNum()))
            linker_idxs.append(at.GetIdx()+num)
    return data(mol=mol, smi=smi, aidx_map_lidx=aidx_map_lidx), linker_idxs

def combine_mols(mols: Any) -> Any:
    combo = Chem.CombineMols(mols[0], mols[1])
    if len(mols) >= 3:
        for mol in mols[2:]:
            combo = Chem.CombineMols(combo, mol)
    return combo

def get_linker_info(smis: Any) -> Any:
    index_map_info, mols, all_linker_idxs = [], [], []
    num = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        res, linker_idxs = get_linker_map_index(mol, smi, num)
        num = num + mol.GetNumAtoms()
        index_map_info.append(res)
        mols.append(mol)
        all_linker_idxs.extend(linker_idxs)
    return index_map_info, mols, all_linker_idxs

def get_link_atom(index_map_info: Any, max_linker_idx: Any) -> Any:
    link_atom_pair = {}
    for idx in range(1, max_linker_idx+1):
        link_atom_pair.update({idx:[]})
        
    for res in index_map_info:
        for aidx_lidx in res.aidx_map_lidx:
            link_atom_pair[aidx_lidx.linker_idx].append(aidx_lidx.atom_idx)
    return link_atom_pair

def remove_linker(emol: Any, lidxs: Any) -> Any:
    slidxs = sorted(lidxs, reverse=True)
    for idx in slidxs:
        emol.RemoveAtom(idx)
    return emol

def get_max_linker_idx(smis: Any) -> Any:
    linker_idxs = []
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        for at in m.GetAtoms():
            if at.GetSymbol() == "*":
                linker_idxs.append(at.GetAtomMapNum())
    return max(linker_idxs)
              
def link_fragment(smis: Any) -> Any:
    index_map_info, mols, all_linker_idxs = get_linker_info(smis)
    max_linker_idx = get_max_linker_idx(smis)
    link_atom_pair = get_link_atom(index_map_info, max_linker_idx)

    combo = combine_mols(mols)
    ecombo = Chem.EditableMol(combo)
    for lidx, atom_pair in link_atom_pair.items():
        if len(atom_pair) == 0:
            continue
        idx0, idx1 = sorted(atom_pair)
        ecombo.AddBond(idx0, idx1, order=Chem.rdchem.BondType.SINGLE)
    ecombo = remove_linker(ecombo, all_linker_idxs)
    nmol = ecombo.GetMol()
    Chem.SanitizeMol(nmol)
    return nmol

if __name__=="__main__":
    smi1 = '[*:3]c1cc(C([*:1])([*:2])F)cc(C2(C(C)C)C(C#N)=C(N)Oc3[nH]nc(C)c32)c1'
    smi2 = '[*:1]F'
    smi3 = '[*:2]F'
    smi4 = '[*:3]N1CCCC1'
    smis = [smi3, smi2, smi1, smi4]
    m = link_fragment(smis)
    print(Chem.MolToSmiles(m))
