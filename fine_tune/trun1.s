#! /bin/bash

#SBATCH --job-name=fine-tuning
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=log.out

module purge


singularity exec --nv --overlay  /scratch/projects/yzlab/xp2042/overlay-script-v3-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u fine_tuning.py fold_1 > log1.o 2>&1"

singularity exec --nv --overlay  /scratch/projects/yzlab/xp2042/overlay-script-v3-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u fine_tuning.py fold_2 > log2.o 2>&1"


singularity exec --nv --overlay  /scratch/projects/yzlab/xp2042/overlay-script-v3-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u fine_tuning.py fold_3 > log3.o 2>&1"

singularity exec --nv --overlay  /scratch/projects/yzlab/xp2042/overlay-script-v3-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u fine_tuning.py fold_4 > log4.o 2>&1"


singularity exec --nv --overlay  /scratch/projects/yzlab/xp2042/overlay-script-v3-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate pyg; python -u fine_tuning.py fold_5 > log5.o 2>&1"




