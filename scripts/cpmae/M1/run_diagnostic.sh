#!/bin/bash
# M1: Diagnostic Study — measure policy sensitivity to visual degradation
#
# Trains ACT and DP with various image degradations on libero_spatial (10 tasks).
# Uses best hyperparameters from M0+ sweep.
#
# Phase-aware degradations require pre-computed contact labels:
#   python scripts/cpmae/contact_detector.py --all_tasks --output_dir=results/contact_labels
#
# Usage:
#   bash scripts/cpmae/run_diagnostic.sh 0 1 2 3   # GPU indices
#   bash scripts/cpmae/run_diagnostic.sh 0          # single GPU
#
# Estimated: ~200 GPU-hours

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}

# Best hyperparameters from M0+ (update after sweep)
ACT_BS=8
ACT_LR=1e-5
DP_BS=8
DP_LR=1e-4

STEPS=100000
EVAL_FREQ=10000
SAVE_FREQ=50000
N_EVAL_EPISODES=20
EVAL_BATCH=20
SEED=42
RESULTS_DIR="results/M1_diagnostic"
REPO_ID="HuggingFaceVLA/libero"
CONTACT_LABELS_DIR="results/contact_labels"
SUITE="libero_spatial"

mkdir -p "$RESULTS_DIR"

# Ensure contact labels exist for phase-aware degradations
if [ ! -d "$CONTACT_LABELS_DIR" ] || [ -z "$(ls -A $CONTACT_LABELS_DIR 2>/dev/null)" ]; then
  echo "Generating contact labels for libero_spatial..."
  python3 scripts/cpmae/contact_detector.py --all_tasks --output_dir="$CONTACT_LABELS_DIR"
fi

# Trap Ctrl+C
cleanup() {
  echo ""; echo "Caught interrupt — killing background jobs..."
  kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; exit 1
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

# Helper: resolve env task_id for a dataset task_index
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
for suite_name in ['$SUITE']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i in range(len(suite.tasks)):
        if suite.get_task(i).language.strip().lower() == task_name:
            sys.stdout = _real; print(i); exit()
sys.stdout = _real; print(-1)
"
}

# Get task indices for the suite
TASK_INDICES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark
import os, sys, io
os.environ['LIBERO_QUIET'] = '1'
_real = sys.stdout; sys.stdout = io.StringIO()
suite = benchmark.get_benchmark_dict()['$SUITE']()
meta = LeRobotDatasetMetadata('$REPO_ID')
ds_tasks = {name.strip().lower(): int(row['task_index']) for name, row in meta.tasks.iterrows()}
sys.stdout = _real
indices = []
for i in range(len(suite.tasks)):
    task_name = suite.get_task(i).language.strip().lower()
    if task_name in ds_tasks:
        indices.append(ds_tasks[task_name])
print(' '.join(str(x) for x in sorted(indices)))
")
echo "Suite $SUITE task_indices: $TASK_INDICES"

# Define diagnostic experiments
# Format: RUN_ID POLICY DEGRADE_NAME
EXPERIMENTS=(
  # M1a: Uniform degradation - ACT
  "R010 act none"
  "R011 act blur_sigma_2"
  "R012 act blur_sigma_5"
  "R013 act grayscale"
  "R014 act low_res_64"
  "R015 act low_res_32"
  # M1a: Uniform degradation - DP
  "R020 dp none"
  "R021 dp blur_sigma_2"
  "R022 dp blur_sigma_5"
  "R023 dp grayscale"
  "R024 dp low_res_64"
  "R025 dp low_res_32"
  # M1b: Phase-aware degradation - ACT
  "R030 act degrade_contact_blur5"
  "R031 act degrade_transit_blur5"
  "R032 act degrade_contact_lowres32"
  "R033 act degrade_transit_lowres32"
  # M1b: Phase-aware degradation - DP
  "R034 dp degrade_contact_blur5"
  "R035 dp degrade_transit_blur5"
  "R036 dp degrade_contact_lowres32"
  "R037 dp degrade_transit_lowres32"
)

# For each experiment, run across all tasks in the suite
# We process one (experiment, task) pair per GPU slot
jobs=()
for exp in "${EXPERIMENTS[@]}"; do
  read -r run_id policy degrade <<< "$exp"
  for task_idx in $TASK_INDICES; do
    jobs+=("$run_id $policy $degrade $task_idx")
  done
done

echo "Total jobs: ${#jobs[@]} (${#EXPERIMENTS[@]} experiments x $(echo $TASK_INDICES | wc -w) tasks)"
echo "Running on $NUM_GPUS GPUs"
echo ""

run_one_job() {
  local run_id=$1 policy=$2 degrade=$3 task_idx=$4 gpu=$5

  local episodes=$(resolve_episodes "$task_idx")
  local env_task_id=$(resolve_env_task "$task_idx")
  local run_dir="$RESULTS_DIR/${run_id}_${policy}_${degrade}/task_${task_idx}"

  if [ -d "$run_dir/checkpoints/last/pretrained_model" ]; then
    echo "[GPU $gpu] $run_id $policy $degrade task=$task_idx — completed, skipping"
    return 0
  fi

  echo "[GPU $gpu] $run_id $policy $degrade task=$task_idx"

  local policy_type bs lr
  if [ "$policy" = "act" ]; then
    policy_type="act"
    bs=$ACT_BS
    lr=$ACT_LR
  else
    policy_type="diffusion"
    bs=$DP_BS
    lr=$DP_LR
  fi

  local base_train_args=(
    --dataset.repo_id=$REPO_ID
    --dataset.episodes="$episodes"
    --policy.type=$policy_type
    --env.type=libero
    --env.task=$SUITE
    --env.task_ids="[$env_task_id]"
    --batch_size=$bs
    --steps=$STEPS
    --eval_freq=$EVAL_FREQ
    --save_freq=$SAVE_FREQ
    --eval.n_episodes=$N_EVAL_EPISODES
    --eval.batch_size=$EVAL_BATCH
    --seed=$SEED
    --output_dir="$run_dir"
    --job_name="${run_id}_task${task_idx}"
    --wandb.enable=true
    --wandb.project=cpmae_diagnostic
    --policy.push_to_hub=false
  )

  if [ "$policy_type" = "act" ]; then
    base_train_args+=(
      --policy.vision_backbone=resnet18
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
      --policy.freeze_backbone=true
      --policy.optimizer_lr=$lr
      --policy.optimizer_lr_backbone=$lr
    )
  else
    base_train_args+=(
      --policy.vision_backbone=resnet18
      --policy.optimizer_lr=$lr
    )
  fi

  if [ "$degrade" = "none" ]; then
    # Standard lerobot-train (no degradation)
    CUDA_VISIBLE_DEVICES=$gpu lerobot-train "${base_train_args[@]}"
  else
    # Use our degradation wrapper
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/cpmae/train_with_degradation.py \
      --degrade_name="$degrade" \
      --contact_labels_dir="$CONTACT_LABELS_DIR" \
      -- "${base_train_args[@]}"
  fi
}

# Dispatch jobs across GPUs
FAILED_JOBS=0
i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  job_ids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break

    read -r run_id policy degrade task_idx <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}

    run_one_job "$run_id" "$policy" "$degrade" "$task_idx" "$gpu" &
    pids+=($!)
    job_ids+=("$run_id $policy $degrade $task_idx")
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
echo "M1 Diagnostic Study Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
if [ $FAILED_JOBS -gt 0 ]; then
  echo "WARNING: $FAILED_JOBS job(s) failed — check logs above"
fi
echo ""
echo "Collect results:"
echo "  python scripts/cpmae/collect_results.py --input_dir=$RESULTS_DIR --milestone=M1"

exit $( [ $FAILED_JOBS -gt 0 ] && echo 1 || echo 0 )
