from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina


def cluster_conformers( mol,  rmsd_cutoff=0.5) :
    n_conf = mol.GetNumConformers()
    if n_conf < 2:
        return [[conf.GetId() for conf in mol.GetConformers()]]

    dists = []
    num_conformers = mol.GetNumConformers()
    for i in range(num_conformers):
        for j in range(i):
            dists.append(rdMolAlign.GetBestRMS(mol, mol, i, j))
    clusters = Butina.ClusterData(dists, num_conformers, rmsd_cutoff, isDistData=True, reordering=True)
    return clusters


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
    
    clusters = cluster_conformers(mol, rmsd_cutoff=0.5)

    confs = []
    for cid in clusters:
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
