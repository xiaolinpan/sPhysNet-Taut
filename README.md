# Fast and Accurate Prediction of Tautomer Ratios in Aqueous Solutions via Siamese Network

Tautomeric equilibrium is pivotal in chemical and biological processes, influencing molecular stability, reactivity, biological activity, and ADME-Tox properties. Many drug-like molecules exist in multiple tautomeric states in aqueous solutions, making the rapid and accurate prediction of tautomer ratios and the the identification of major species essential for computational drug discovery. Over recent decades, numerous methods based on empirical rules and quantum mechanical (QM) calculations have been developed to predict tautomer ratios in various solvents. In this work, we present sPhysNet-siamese, a deep learning model fine-tuned on experimental data from a pre-trained model based on siamese network protocol, designed to predict tautomer ratios in aqueous solutions using MMFF94-optimized geometries directly. To train the pre-trained model, we create the Frag20-Taut dataset, calculated at the B3LYP/6-31G* level with the SMD solvation model, containing \~1 million molecular conformations with electronic energies in both gas and water phases. Our results demonstrate that sPhysNet-siamese excels in predicting tautomer ratios in aqueous solutions. On an experimental test set, sPhysNet-siamese surpasses all other methods, achieving state-of-the-art performance with an RMSE of 1.90 kcal/mol on the test set and an RMSE of 1.40 kcal/mol on the SAMPL2 challenge. Furthermore, we developed a user-friendly tool to predict stable tautomeric states in aqueous solutions, which enumerates all possible tautomeric states and ranks them using the sPhysNet-siamese model.

---
<div align="center">
    <img src="https://github.com/xiaolinpan/sPhysNet-Taut/blob/main/images/TOC.png" alt="image" width="450"/>
</div>

## Requirements

* Python
* numpy
* RDKit
* scipy
* pandas
* pytorch
* pytorch geometric
* torch-scatter
* torch-sparse

## Usage

```
python predict_tautomer.py --help

usage: predict_tautomer.py [-h] [--smi SMI]
                           [--low_energy_tautomer_cutoff LOW_ENERGY_TAUTOMER_CUTOFF]
                           [--cutmol CUTMOL] [--num_confs NUM_CONFS] [--ph PH]
                           [--tph TPH] [--output OUTPUT]

calculate low-energy tautomer for small molecules

optional arguments:
  -h, --help            show this help message and exit
  --smi SMI             the molecular smiles
  --low_energy_tautomer_cutoff LOW_ENERGY_TAUTOMER_CUTOFF
                        the energy cutoff for low energy
  --cutmol CUTMOL       determine to frag the molecule
  --num_confs NUM_CONFS
                        the number of conformation for solvation energy
                        prediction
  --ph PH               the target pH for protonation states generation
  --tph TPH             pH tolerance for protonation states generation
  --output OUTPUT       the output SDF file name

```
These is a example for the MolTaut usage, the ligand is extracted from pdbid 5v7i:
```
python predict_tautomer.py --smi "Cc1c2c([nH]n1)OC(=C([C@@]2(c3cc(cc(c3)N4CCCC4)C(F)(F)F)C(C)C)C#N)N"
```

## Citation
If you use this project in your research, please cite the following papers:

1.**MolGpka: A Web Server for Small Molecule pKa Prediction Using a Graph-Convolutional Neural Network**
Authors: Xiaolin Pan, Hao wang, et al.
Journal: JCIM
Year: 2021-07-12
DOI: https://doi.org/10.1021/acs.jcim.1c00075
 
2.
3.
