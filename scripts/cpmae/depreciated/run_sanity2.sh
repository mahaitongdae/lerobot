#!/bin/bash
# M0 Sanity Check Round 2 — stronger hyperparameters
#
# R003: ACT + ResNet18 ImageNet unfrozen, bs=32, lr=1e-4, 10k steps
# R004: DP + ResNet18, bs=32, lr=1e-4, 10k steps
#
# Usage:
#   bash scripts/cpmae/run_sanity2.sh [GPU0_ID [GPU1_ID]]

set -euo pipefail

export PYTHONWARNINGS="ignore"
export PYOPENGL_PLATFORM=egl

GPU0=${1:-0}
GPU1=${2:-1}

cleanup() {
  echo ""
  echo "Caught interrupt — killing all background jobs..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "All jobs stopped."
  exit 1
}
trap cleanup SIGINT SIGTERM

STEPS=10000
EVAL_FREQ=1000
SAVE_FREQ=5000
N_EVAL_EPISODES=20
EVAL_BATCH=10
TASK_INDEX=0
SEED=42
RESULTS_DIR="results/M0_sanity"
REPO_ID="HuggingFaceVLA/libero"

mkdir -p "$RESULTS_DIR"

# Resolve episodes for task 0
EPISODES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('$REPO_ID')
task_name = meta.tasks[meta.tasks['task_index'] == $TASK_INDEX].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
")
echo "Task $TASK_INDEX episodes: $EPISODES"

# Resolve env suite and task_id
read -r SUITE ENV_TASK_ID <<< $(python3 -c "
import sys, io, os
os.environ['LIBERO_QUIET'] = '1'
_real = sys.stdout; sys.stdout = io.StringIO()
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark
meta = LeRobotDatasetMetadata('$REPO_ID')
ds_tasks = {int(row['task_index']): name for name, row in meta.tasks.iterrows()}
task_name = ds_tasks[$TASK_INDEX].strip().lower()
for suite_name in ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i in range(len(suite.tasks)):
        if suite.get_task(i).language.strip().lower() == task_name:
            sys.stdout = _real; print(suite_name, i); exit()
sys.stdout = _real; print('NOT_FOUND -1')
")
echo "Env: suite=$SUITE task_id=$ENV_TASK_ID"

echo ""
echo "========================================="
echo "R003: ACT + ResNet18 ImageNet (unfrozen), bs=32, lr=1e-4"
echo "R004: DP + ResNet18, bs=32, lr=1e-4"
echo "Running in parallel on GPU$GPU0 and GPU$GPU1"
echo "========================================="

R003_DIR="$RESULTS_DIR/R003_act_resnet18_unfrozen_bs32_lr1e-4"
R004_DIR="$RESULTS_DIR/R004_dp_resnet18_bs32_lr1e-4"
R003_LOG="$RESULTS_DIR/R003.log"
R004_LOG="$RESULTS_DIR/R004.log"

if [ ! -d "$R003_DIR" ]; then
  MUJOCO_EGL_DEVICE_ID=$GPU0 CUDA_VISIBLE_DEVICES=$GPU0 lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$EPISODES" \
    --policy.type=act \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.freeze_backbone=false \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="[$ENV_TASK_ID]" \
    --batch_size=32 \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$SEED \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=1e-4 \
    --output_dir="$R003_DIR" \
    --job_name=R003_act_resnet18_unfrozen_bs32_lr1e-4 \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    > "$R003_LOG" 2>&1 &
  PID_R003=$!
  echo "R003 started (PID=$PID_R003) on GPU$GPU0 — log: $R003_LOG"
else
  echo "R003 already exists, skipping"
  PID_R003=""
fi

if [ ! -d "$R004_DIR" ]; then
  MUJOCO_EGL_DEVICE_ID=$GPU1 CUDA_VISIBLE_DEVICES=$GPU1 lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$EPISODES" \
    --policy.type=diffusion \
    --policy.vision_backbone=resnet18 \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="[$ENV_TASK_ID]" \
    --batch_size=32 \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$SEED \
    --policy.optimizer_lr=1e-4 \
    --output_dir="$R004_DIR" \
    --job_name=R004_dp_resnet18_bs32_lr1e-4 \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    > "$R004_LOG" 2>&1 &
  PID_R004=$!
  echo "R004 started (PID=$PID_R004) on GPU$GPU1 — log: $R004_LOG"
else
  echo "R004 already exists, skipping"
  PID_R004=""
fi

FAIL=0
if [ -n "$PID_R003" ]; then
  wait $PID_R003 && echo "R003 DONE" || { echo "R003 FAILED (see $R003_LOG)"; FAIL=1; }
fi
if [ -n "$PID_R004" ]; then
  wait $PID_R004 && echo "R004 DONE" || { echo "R004 FAILED (see $R004_LOG)"; FAIL=1; }
fi

echo ""
echo "========================================="
echo "M0 Sanity Check Round 2 Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
if [ $FAIL -ne 0 ]; then
  echo "WARNING: Some runs failed — check logs"
  exit 1
fi
