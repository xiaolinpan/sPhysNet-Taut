#! /bin/bash

#SBATCH --job-name=tautomer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=72:00:00
#SBATCH --output=log.out

module purge

singularity exec --nv --overlay  /scratch/projects/yzlab/xp2042/overlay-jp-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u ranking_tautomer_example.py >log.o 2>&1"
