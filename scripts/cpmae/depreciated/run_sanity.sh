#!/bin/bash
# M0: Sanity Check — verify training + eval pipeline works end-to-end
#
# Runs:
#   R000: ACT + ResNet18 ImageNet frozen, libero_spatial task 0, 10k steps
#   R001: DP + ResNet18, libero_spatial task 0, 10k steps
#   R002: Contact labeling check
#
# Usage:
#   bash scripts/cpmae/run_sanity.sh [GPU0_ID [GPU1_ID]]
#
# R000 and R001 run in parallel on GPU0 and GPU1 respectively.
# Expected: ~1 hour (parallel on two GPUs)

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

# Resolve env suite and task_id for evaluation
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
echo "R000: ACT + ResNet18 ImageNet (frozen)"
echo "R001: DP + ResNet18"
echo "Running in parallel on GPU$GPU0 and GPU$GPU1"
echo "========================================="

R000_DIR="$RESULTS_DIR/R000_act_resnet18_frozen"
R001_DIR="$RESULTS_DIR/R001_dp_resnet18"
R000_LOG="$RESULTS_DIR/R000.log"
R001_LOG="$RESULTS_DIR/R001.log"

if [ ! -d "$R000_DIR" ]; then
  MUJOCO_EGL_DEVICE_ID=$GPU0 CUDA_VISIBLE_DEVICES=$GPU0 lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$EPISODES" \
    --policy.type=act \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.freeze_backbone=true \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="[$ENV_TASK_ID]" \
    --batch_size=8 \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$SEED \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_lr_backbone=1e-5 \
    --output_dir="$R000_DIR" \
    --job_name=R000_act_resnet18_frozen \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    > "$R000_LOG" 2>&1 &
  PID_R000=$!
  echo "R000 started (PID=$PID_R000) on GPU$GPU0 — log: $R000_LOG"
else
  echo "R000 already exists, skipping"
  PID_R000=""
fi

if [ ! -d "$R001_DIR" ]; then
  MUJOCO_EGL_DEVICE_ID=$GPU1 CUDA_VISIBLE_DEVICES=$GPU1 lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$EPISODES" \
    --policy.type=diffusion \
    --policy.vision_backbone=resnet18 \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="[$ENV_TASK_ID]" \
    --batch_size=8 \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$SEED \
    --output_dir="$R001_DIR" \
    --job_name=R001_dp_resnet18 \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    > "$R001_LOG" 2>&1 &
  PID_R001=$!
  echo "R001 started (PID=$PID_R001) on GPU$GPU1 — log: $R001_LOG"
else
  echo "R001 already exists, skipping"
  PID_R001=""
fi

# Wait for both jobs and check exit codes
FAIL=0
if [ -n "$PID_R000" ]; then
  wait $PID_R000 && echo "R000 DONE" || { echo "R000 FAILED (see $R000_LOG)"; FAIL=1; }
fi
if [ -n "$PID_R001" ]; then
  wait $PID_R001 && echo "R001 DONE" || { echo "R001 FAILED (see $R001_LOG)"; FAIL=1; }
fi
[ $FAIL -eq 0 ] || exit 1

# echo ""
# echo "========================================="
# echo "R002: Contact labeling check"
# echo "========================================="

# python3 scripts/cpmae/contact_detector.py \
#   --task_index=$TASK_INDEX \
#   --window=5 \
#   --output_dir="$RESULTS_DIR/contact_labels"

echo ""
echo "========================================="
echo "M0 Sanity Check Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
