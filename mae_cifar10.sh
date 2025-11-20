#!/bin/bash
#SBATCH --job-name=mae_cifar10
#SBATCH --output=logs/slurm/%x_%j.out   # Log output to logs/<job-name>_<job-id>.out
#SBATCH --error=logs/slurm/%x_%j.err    # Error logs (optional)
#SBATCH --time=12:00:00            # Max walltime
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=128G                 # Memory


# Load your environment
module purge
source ~/.bashrc
conda activate devinterp_env

# Run training script with Hydra config
srun -c 4 python3 pretrain_mae.py \
    --batch_size=128 \

