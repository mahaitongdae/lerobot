#!/bin/bash
# M0+: Hyperparameter sweep — find best (batch_size, lr) for ACT and DP
#
# Grid: batch_size in {8, 16, 32} x lr in {1e-4, 5e-5, 1e-5} = 9 configs per policy
# Tasks: task_index in {0, 10, 20, 30} — spanning different LIBERO suites
# Total: 18 configs x 4 tasks = 72 runs
#
# Usage:
#   bash scripts/cpmae/run_hp_sweep.sh 0 1 2 3    # GPU indices
#   bash scripts/cpmae/run_hp_sweep.sh 0           # single GPU (sequential)
#
# Estimated: ~240 GPU-hours total

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}

STEPS=100000
EVAL_FREQ=10000
SAVE_FREQ=20000
N_EVAL_EPISODES=20
EVAL_BATCH=10
SEED=42
RESULTS_DIR="results/M0_deep"
REPO_ID="HuggingFaceVLA/libero"

BATCH_SIZES=(16 32 64)
LRS=(1e-4 5e-5 1e-5)
TASK_INDICES=(0 10 20 30)

mkdir -p "$RESULTS_DIR"

# Trap Ctrl+C to kill all background jobs
cleanup() {
  echo ""; echo "Caught interrupt — killing background jobs..."
  kill $(jobs -p) 2>/dev/null; wait 2>/dev/null
  echo "All jobs stopped."; exit 1
}
trap cleanup SIGINT SIGTERM

# Helper: resolve episodes for a task
resolve_episodes() {
  local task_idx=$1
  python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('$REPO_ID')
task_name = meta.tasks[meta.tasks['task_index'] == $task_idx].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
"
}

# Helper: resolve env suite and task_id for a dataset task_index
resolve_env_task() {
  local task_idx=$1
  python3 -c "
import sys, io, os
os.environ['LIBERO_QUIET'] = '1'
_real = sys.stdout; sys.stdout = io.StringIO()
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark
meta = LeRobotDatasetMetadata('$REPO_ID')
ds_tasks = {int(row['task_index']): name for name, row in meta.tasks.iterrows()}
task_name = ds_tasks[$task_idx].strip().lower()
for suite_name in ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i in range(len(suite.tasks)):
        if suite.get_task(i).language.strip().lower() == task_name:
            sys.stdout = _real; print(suite_name, i); exit()
sys.stdout = _real; print('NOT_FOUND -1')
"
}

# Pre-resolve episodes and env info for each task
declare -A TASK_EPISODES TASK_SUITES TASK_ENV_IDS
for task_idx in "${TASK_INDICES[@]}"; do
  TASK_EPISODES[$task_idx]=$(resolve_episodes "$task_idx")
  read -r suite env_id <<< "$(resolve_env_task "$task_idx")"
  TASK_SUITES[$task_idx]=$suite
  TASK_ENV_IDS[$task_idx]=$env_id
  echo "Task $task_idx -> suite=${TASK_SUITES[$task_idx]}, env_task_id=${TASK_ENV_IDS[$task_idx]}, episodes=${TASK_EPISODES[$task_idx]}"
done

# Build job list: (policy, batch_size, lr, task_idx)
jobs=()
for task_idx in "${TASK_INDICES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for lr in "${LRS[@]}"; do
      jobs+=("act $bs $lr $task_idx")
    done
  done
  for bs in "${BATCH_SIZES[@]}"; do
    for lr in "${LRS[@]}"; do
      jobs+=("dp $bs $lr $task_idx")
    done
  done
done

echo ""
echo "Total sweep jobs: ${#jobs[@]} (18 configs x ${#TASK_INDICES[@]} tasks) across $NUM_GPUS GPUs"
echo ""

FAILED_JOBS=0
i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  job_ids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break

    read -r policy bs lr task_idx <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}

    EPISODES="${TASK_EPISODES[$task_idx]}"
    SUITE="${TASK_SUITES[$task_idx]}"
    ENV_TASK_ID="${TASK_ENV_IDS[$task_idx]}"

    if [ "$policy" = "act" ]; then
      RUN_NAME="R000_act_bs${bs}_lr${lr}_task${task_idx}"
      RUN_DIR="$RESULTS_DIR/$RUN_NAME"

      if [ -d "$RUN_DIR/checkpoints/last/pretrained_model" ]; then
        echo "[GPU $gpu] $RUN_NAME — completed, skipping"
        continue
      fi

      echo "[GPU $gpu] $RUN_NAME (ACT, bs=$bs, lr=$lr, task=$task_idx)"
      CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
        --dataset.repo_id=$REPO_ID \
        --dataset.episodes="$EPISODES" \
        --policy.type=act \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.freeze_backbone=true \
        --env.type=libero \
        --env.task=$SUITE \
        --env.task_ids="[$ENV_TASK_ID]" \
        --batch_size=$bs \
        --steps=$STEPS \
        --eval_freq=$EVAL_FREQ \
        --save_freq=$SAVE_FREQ \
        --eval.n_episodes=$N_EVAL_EPISODES \
        --eval.batch_size=$EVAL_BATCH \
        --seed=$SEED \
        --policy.optimizer_lr=$lr \
        --policy.optimizer_lr_backbone=$lr \
        --output_dir="$RUN_DIR" \
        --job_name="$RUN_NAME" \
        --wandb.enable=true \
        --wandb.project=cpmae_hp_sweep \
        --policy.push_to_hub=false &
      pids+=($!)
      job_ids+=("$RUN_NAME")

    elif [ "$policy" = "dp" ]; then
      RUN_NAME="R001_dp_bs${bs}_lr${lr}_task${task_idx}"
      RUN_DIR="$RESULTS_DIR/$RUN_NAME"

      if [ -d "$RUN_DIR/checkpoints/last/pretrained_model" ]; then
        echo "[GPU $gpu] $RUN_NAME — completed, skipping"
        continue
      fi

      echo "[GPU $gpu] $RUN_NAME (DP, bs=$bs, lr=$lr, task=$task_idx)"
      CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
        --dataset.repo_id=$REPO_ID \
        --dataset.episodes="$EPISODES" \
        --policy.type=diffusion \
        --policy.vision_backbone=resnet18 \
        --env.type=libero \
        --env.task=$SUITE \
        --env.task_ids="[$ENV_TASK_ID]" \
        --batch_size=$bs \
        --steps=$STEPS \
        --eval_freq=$EVAL_FREQ \
        --save_freq=$SAVE_FREQ \
        --eval.n_episodes=$N_EVAL_EPISODES \
        --eval.batch_size=$EVAL_BATCH \
        --seed=$SEED \
        --policy.optimizer_lr=$lr \
        --output_dir="$RUN_DIR" \
        --job_name="$RUN_NAME" \
        --wandb.enable=true \
        --wandb.project=cpmae_hp_sweep \
        --policy.push_to_hub=false &
      pids+=($!)
      job_ids+=("$RUN_NAME")
    fi
  done

  for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then
      echo "FAILED: ${job_ids[$j]}"
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
  done

  i=$((i + NUM_GPUS))
done

echo ""
echo "========================================="
echo "M0+ HP Sweep Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
if [ $FAILED_JOBS -gt 0 ]; then
  echo "WARNING: $FAILED_JOBS job(s) failed — check logs above"
fi
echo ""
echo "Run collect_results.py to find best hyperparameters:"
echo "  python scripts/cpmae/collect_results.py --input_dir=$RESULTS_DIR --output=results/M0_deep_summary.json"

exit $( [ $FAILED_JOBS -gt 0 ] && echo 1 || echo 0 )
