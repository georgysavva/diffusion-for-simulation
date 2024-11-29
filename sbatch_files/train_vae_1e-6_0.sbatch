#!/bin/bash

#SBATCH --constraint='a100|h100'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_vae_1e-6_0
#SBATCH --output=slurm_outs/%j.out

module purge

singularity exec --nv \
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl11330/diffusion-for-simulation; conda activate ./penv; \
        python src/vae/train_vae.py --tag 1128_0_train_vae_lr1e-6 --learning_rate 1e-6 --wandb"
