#!/bin/bash

#SBATCH --constraint='a100|h100'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_vanilla_dit_baseline
#SBATCH --output=slurm_outs/%j.out

module purge;
module load anaconda3/2024.02;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate ./penv;

python scripts/evaluate_trajectory.py --model_version diffusion_model_epoch_00131.pt --run_dir /scratch/gs4288/shared/diffusion_for_simulation/runs/spring-oath-23
