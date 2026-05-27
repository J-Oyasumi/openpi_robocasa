#!/bin/bash
#SBATCH --job-name=pi05_rc_ft
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                # 2x B200; adjust to cluster (e.g. --gpus=2 / --gpus-per-node=2)
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.log

set -euo pipefail

# ---- edit these ----
CONFIG=pi05_robocasa_seen_pretrain          # finetune config name
EXP_NAME=ft_2xb200_$(date +%Y%m%d_%H%M%S)

cd /home/hanjiang/clone/openpi
mkdir -p slurm_logs

# ---- env ----
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openpi                       # env must be able to import robocasa (pip install -e ../robocasa)

# B200 = Blackwell; let XLA use most of the device memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# ---- norm stats (comment out if already computed) ----
srun python scripts/compute_norm_stats.py --config-name=$CONFIG

# ---- train: 2-GPU data parallel (default fsdp_devices=1) ----
# JAX uses both SLURM-allocated GPUs in one process; batch_size must be divisible by 2 (config is 64).
srun python scripts/train.py $CONFIG --exp-name=$EXP_NAME
# If a single GPU runs out of memory, shard across 2 GPUs by adding: --fsdp_devices=2
