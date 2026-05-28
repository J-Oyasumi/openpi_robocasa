#!/bin/bash
#SBATCH --job-name=pi05_rc_ft
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.log

srun python scripts/train.py pi05_robocasa_seen_pretrain --exp-name=ft_2xb200
