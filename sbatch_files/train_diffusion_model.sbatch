#!/bin/bash

#SBATCH --constraint='a100|h100|rtx8000'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_vanilla_dit_baseline
#SBATCH --output=slurm_outs/%j.out

module purge;


singularity exec --nv \
    --overlay /scratch/gs4288/singularity_envs/diffusion-for-simulation.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/gs4288/repos/diffusion-for-simulation; conda activate dfsenv; \
        python src/main.py experiment_name=baseline"

