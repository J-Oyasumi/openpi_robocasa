#!/bin/bash
#SBATCH --job-name=pi05_rc_eval
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.log

CKPT=checkpoints/pi05_robocasa_seen_pretrain/ft_2xb200/59999   # checkpoint step dir to eval
PORT=8000
export MUJOCO_GL=egl                                            # offscreen rendering for robocasa

# start inference server in background (own log); client auto-waits until ready
python scripts/serve_policy.py --port=$PORT policy:checkpoint \
  --policy.config=pi05_robocasa_seen_pretrain --policy.dir=$CKPT \
  > slurm_logs/server_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

# eval 1: seen tasks -> target split
python examples/robocasa/main.py --args.port $PORT \
  --args.task_set seen_tasks --args.split target --args.log_dir $CKPT

# eval 2: unseen tasks -> pretrain split
python examples/robocasa/main.py --args.port $PORT \
  --args.task_set unseen_tasks --args.split pretrain --args.log_dir $CKPT

# aggregate success rates
python examples/robocasa/get_eval_stats.py --dir $CKPT

kill $SERVER_PID
