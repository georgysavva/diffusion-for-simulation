#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint='a100|h100'
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=preprocess_data_mod6
#SBATCH --output=slurm_outs/%j_%x.out

module purge;


singularity exec --nv \
    --overlay /scratch/gs4288/singularity_envs/diffusion-for-simulation.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/gs4288/repos/diffusion-for-simulation; conda activate ./penv; \
        python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 6"
