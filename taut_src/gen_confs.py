from rdkit import Chem
from rdkit.Chem import AllChem

def extract_mol_by_confId(mol, confId):
    mol_block = Chem.MolToMolBlock(mol, confId=confId)
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    return mol


def generate_confs(smi, numConfs=1):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    # cids = AllChem.EmbedMultipleConfs(mol, numConfs=128, maxAttempts=1000, numThreads=8)
    
    ps = AllChem.ETKDG()
    ps.maxAttempts = 10000
    ps.randomSeed = 1
    ps.pruneRmsThresh = 0.1
    ps.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs, ps)
    
    confs = []
    for cid in cids:
        mol_conf = extract_mol_by_confId(mol, cid)
        confs.append(mol_conf)
    return confs


def optimize(mol):
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0)
    ff.Initialize()
    ff.Minimize(maxIts=1000)
    E = ff.CalcEnergy()
    return E


def get_low_energy_conf(smi, num_confs, index=0):
    mol_confs = generate_confs(smi, num_confs)
    data = []
    for m in mol_confs:
        E = optimize(m)
        data.append([E, m])
    sdata = sorted(data, key=lambda x: x[0])
    low_energy, opted_conf = sdata[index]
    return opted_conf
