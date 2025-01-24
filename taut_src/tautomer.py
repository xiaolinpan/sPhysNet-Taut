from taut_src.config import transform_path
import copy
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

with open(transform_path, "r") as f:
    conts = f.readlines()
smirks = [line.strip("\n").split("\t") for line in conts]


def uncharge_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    for i in range(5):
        mol = unc.uncharge(mol)
    return Chem.MolToSmiles(mol)


def init_dict(smirks):
    mdict = {}
    for idx, line in enumerate(smirks):
        smrk, name = line
        mdict[str(idx) + "_" + name] = []
    return mdict


def repair_smiles(gm):
    smi = Chem.MolToSmiles(gm)
    new_mol = Chem.MolFromSmiles(smi, sanitize=True)
    if not new_mol:
        #print("generate error smiles for tautomers", Chem.MolToSmiles(gm), smi)
        return
    return Chem.MolToSmiles(new_mol)


def protect_atom(m):
    for at in m.GetAtoms():
        if at == "*":
            at.SetProp('_protected', '1')
    return


def protect_guanidine(mol):
    pattern = Chem.MolFromSmarts("[#7;!R]~[#6;!R](~[#7;!R])~[#7;!R]")
    atom_idx = sum(mol.GetSubstructMatches(pattern), ())
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_idx:
            at.SetProp('_protected', '1')
    return


def protect_amide(mol):
    pattern = Chem.MolFromSmarts("[#7;!R]-[#6;!R]=[#8]")
    atom_idx = sum(mol.GetSubstructMatches(pattern), ())
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_idx:
            at.SetProp('_protected', '1')
    return


def protect_nitro(mol):
    pattern = Chem.MolFromSmarts("[#7](~[#8])~[#8]")
    atom_idx = sum(mol.GetSubstructMatches(pattern), ())
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_idx:
            at.SetProp('_protected', '1')
    return


def protect_nitroso(mol):
    pattern = Chem.MolFromSmarts("[#7]~[#8;X1]")
    atom_idx = sum(mol.GetSubstructMatches(pattern), ())
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_idx:
            at.SetProp('_protected', '1')
    return


def protect_phosphoric(mol):
    pattern = Chem.MolFromSmarts("[#8]~[#15](~[#8])(~[#8])~[#8]")
    atom_idx = sum(mol.GetSubstructMatches(pattern), ())
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_idx:
            at.SetProp('_protected', '1')
    return


def protect_anitro(mol):
    pattern = Chem.MolFromSmarts("[#6]-[#6;!R]=[#7;!R]-[#6]")
    res = [i[1:-1] for i in mol.GetSubstructMatches(pattern)]
    atom_idx = sum(res, ())
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_idx:
            at.SetProp('_protected', '1')
    return

def get_tauts_by_smirks(mm, tauts_dict, kekulize=True):
    m = copy.deepcopy(mm)
    protect_atom(m)
    protect_guanidine(m)
    protect_amide(m)
    protect_nitro(m)
    protect_nitroso(m)
    protect_phosphoric(m)
    protect_anitro(m)

    if kekulize:
        Chem.Kekulize(m, clearAromaticFlags=True)

    for idx, line in enumerate(smirks):
        smrk, name = line
        rxn = AllChem.ReactionFromSmarts(smrk)
        mn = rxn.RunReactants((m,))
        if len(mn) == 0:
            continue
        else:
            for unit in mn:
                gm = unit[0]
                smi = repair_smiles(gm)
                if smi:
                    tauts_dict[str(idx)+"_"+name].append(smi)
    return



def unique_tauts(tauts_dict, m):
    data = namedtuple('tauts',
                      'smi smirks mol')
    tauts_dict.update({"self": [Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m)))]})

    taut_smis = []
    for sidx, tauts in tauts_dict.items():
        for tsmi in set(tauts):
            taut_smis.append(tsmi)
    ntauts_dict = {}
    for tsmi in set(taut_smis):
        ntauts_dict[tsmi] = []
    for sidx, tauts in tauts_dict.items():
        for tsmi in set(tauts):
            ntauts_dict[tsmi].append(sidx)

    ntauts = []
    for taut, sidxs in ntauts_dict.items():
        ntauts.append(
            data(
                smi=taut,
                smirks=sidxs,
                mol=Chem.MolFromSmiles(taut)))
    return ntauts


def filter_kekulize(m, patterns):
    for pattern in patterns:
        matches = sum(m.GetSubstructMatches(pattern), ())
        for at in m.GetAtoms():
            if (at.GetFormalCharge() != 0) and (at.GetIdx() not in matches):
                return False
    return True 


def multi_kekulize(m):
    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mols = Chem.ResonanceMolSupplier(m, Chem.KEKULE_ALL )
    kmols = []
    for m in mols:
        for atom in m.GetAtoms():
             atom.SetIsAromatic(False)
        for bond in m.GetBonds():
             bond.SetIsAromatic(False)
        smi = Chem.MolToSmiles(m, kekuleSmiles=True)
        nm = Chem.MolFromSmiles(smi, ps)
        kmols.append(nm)
    return kmols


def is_include_element(mol, element_list=[15]):
    elements = any([at.GetAtomicNum() in element_list for at in mol.GetAtoms()])
    if elements:
        return True
    else:
        return False


def tauts_for_special_frag(m):
    data = namedtuple('tauts',
                      'smi smirks mol')
    ntauts = []
    ntauts.append(
            data(smi=Chem.MolToSmiles(m),
                 smirks=["self"],
                 mol=m))
    return ntauts


def get_mols_from_dict(tauts_dict):
    all_tsmis = []
    for rule, tsmis in tauts_dict.items():
        if len(tsmis) == 0:
            continue
        for tsmi in tsmis:
            all_tsmis.append(tsmi)
    all_tsmis = set(all_tsmis)
    all_tmols = [Chem.MolFromSmiles(smi) for smi in all_tsmis]
    return all_tmols

def get_tauts_by_dict(tauts_dict):
    all_tmols = get_mols_from_dict(tauts_dict)
    for tm in all_tmols:
        tm = Chem.AddHs(tm)
        try:
            kms = multi_kekulize(tm)

            for km in kms:
                get_tauts_by_smirks(km, tauts_dict)
        except:
            continue
    return 

def enumerate_tauts(om):
    m = copy.deepcopy(om)
    if is_include_element(m):
        ntauts = tauts_for_special_frag(m)
    else:
        tauts_dict = init_dict(smirks)
        m = Chem.AddHs(m)
        get_tauts_by_smirks(m, tauts_dict, kekulize=False)

        try:
            kms = multi_kekulize(m)
        
            for km in kms:
                get_tauts_by_smirks(km, tauts_dict)
        except:
            pass
        
        for i in range(2):
            get_tauts_by_dict(tauts_dict)

        ntauts = unique_tauts(tauts_dict,om)
    return ntauts


if __name__ == "__main__":
    #smi = "Oc1ccccc1"
    #smi = "COC(=O)c1ccc(O)cc1"
    # smi = "N#CC1=C(N)Oc2[nH]ncc2C1"
    #smi = "OSc1ncc[nH]1"
    #smi = "O=C(Cc1ccccc1)c1cccs1"
    #smi = "Oc1nc2ccccc2nc1"
    #smi = "O=C1NN=CN1"
    #smi = "OC(=O)COC(=O)N[C@]12CC[C@H](CC1)[C@@H]1[C@H]2C(=O)N(C1=O)c1ccc(cc1)NC(=O)C"
    #smi = "OCc(n1)cnc(c12)nc(N)[nH]c2=O"
    #smi = "O=[N+]([O-])c1ccc2cn[nH]c2c1"
    #smi = "c1ccc2cn[nH]c2c1"
    smi = "Nc1nc(O)c2[nH]nnc2n1"
    m = Chem.MolFromSmiles(smi)
    ms = enumerate_tauts(m)
    print(ms)
    print([t.smi for t in ms])
