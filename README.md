# Fast and Accurate Prediction of Tautomer Ratios in Aqueous Solutions via Siamese Network

<p align="justify">
Tautomerization plays a critical role in numerous chemical and biological processes, impacting factors such as molecular stability, reactivity, biological activity, and ADME-Tox properties. Many drug-like molecules exist in multiple tautomeric states in aqueous solutions and complicating drug discovery. Predicting these tautomeric ratios and identifying the predominant species rapidly and accurately is crucial for computational drug discovery. In this study, we introduce sPhysNet-Taut, a deep learning model fine-tuned with experimental data leveraging the Siamese network, built upon a pre-trained model. This model predicts tautomer ratios in aqueous solutions using MMFF94-optimized geometries directly. We demonstrate how to correct the limitation of the pre-trained model for predicting molecular internal energies and solvent effects by fine-tuning using experimental data. On an experimental test set, sPhysNet-Taut surpasses all other methods, achieving state-of-the-art performance with an RMSE of 1.9 kcal/mol on the 100-tautomer set and an RMSE of 1.0 kcal/mol on the SAMPL2 challenge, and providing the best ranking power for tautomer pairs. This work not only provides a useful deep learning model for predicting tautomer ratios, but also provides a protocol for modeling pairwise data. To facilitate user-friendliness, we developed a readily accessible tool to predict stable tautomeric states in aqueous solutions, enumerating all possible tautomeric states and ranking them using the sPhysNet-Taut model.
</p>

---
<div align="center">
    <img src="https://github.com/xiaolinpan/sPhysNet-Taut/blob/main/images/TOC.png" alt="image" width="450"/>
</div>

## Requirements

* Python 3.10.13
* numpy 1.26.3
* RDKit 2024.03.3
* scipy 
* pandas 2.1.4
* pytorch 2.2.0
* pytorch geometric 2.4.0
* torch-scatter 2.1.2
* torch-sparse 0.6.18
* torch-cluster 1.6.3
* torch-spline-conv 1.2.2
* treelib 1.6.1
* mols2grid 2.0.0

## Usage

```
usage: predict_tautomer.py [-h] [--smi SMI]
                           [--low_energy_tautomer_cutoff LOW_ENERGY_TAUTOMER_CUTOFF]
                           [--num_confs NUM_CONFS]
                           [--ionization IONIZATION] [--ph PH] [--tph TPH]
                           [--output OUTPUT]

To calculate low-energy tautomeric states for small molecules by a deep learning model.

options:
  -h, --help            show this help message and exit
  --smi SMI             the molecular smiles
  --low_energy_tautomer_cutoff LOW_ENERGY_TAUTOMER_CUTOFF
                        the energy cutoff for low energy
  --num_confs NUM_CONFS
                        the number of conformation for solvation energy
                        prediction
  --ionization IONIZATION
                        determine to generate ionization states by predicted pKa
                        using the given pH
  --ph PH               the target pH for protonation states generation
  --tph TPH             pH tolerance for protonation states generation
  --output OUTPUT       the output SDF file name

```
These is a example for the tautomeric states prediction:
```
python predict_tautomer.py --smi "Cc1c2c([nH]n1)OC(=C([C@@]2(c3cc(cc(c3)N4CCCC4)C(F)(F)F)C(C)C)C#N)N"
```

Alternatively, you can run it via Jupyter Notebook; the results will be displayed using Mols2Grid with energies. Additionally, we provide a free and user-friendly web server for the community, which you can access at [https://yzhang.hpc.nyu.edu/tautomer](https://yzhang.hpc.nyu.edu/tautomer).

## Citation
If you use this project in your research, please cite the following papers:

1. Xia, S., Zhang, D., & Zhang, Y. (2023). **Multitask Deep Ensemble Prediction of Molecular Energetics in Solution: From Quantum Mechanics to Experimental Properties.** Journal of Chemical Theory and Computation, 19(2), 659-668.

2. Pan, X., Wang, H., Li, C., Zhang, J. Z., & Ji, C. (2021). **MolGpka: A Web Server for Small Molecule pKa Prediction Using a Graph-Convolutional Neural Network.** Journal of Chemical Information and Modeling, 61(7), 3159-3165.
 
3. Pan, X., Zhao, F., Zhang, Y., Wang, X., Xiao, X., Zhang, J. Z., & Ji, C. (2023). **MolTaut: A Tool for the Rapid Generation of Favorable Tautomer in Aqueous Solution.** Journal of Chemical Information and Modeling, 63(7), 1833-1840.

4. Dhaked, D. K., Ihlenfeldt, W. D., Patel, H., Delannée, V., & Nicklaus, M. C. (2020). **Toward a comprehensive treatment of tautomerism in chemoinformatics including in InChI V2.** Journal of chemical information and modeling, 60(3), 1253-1275.

## Model architecture

---
<div align="center">
    <img src="https://github.com/xiaolinpan/sPhysNet-Taut/blob/main/images/p6.png" alt="image" width="800"/>
</div>

## Notes

The limitation of our model is that the performance may decrease with increasing molecular size. Since the maximum number of heavy atoms in our DFT-calculated dataset is only **20**, the model may not perform well with larger molecules.
