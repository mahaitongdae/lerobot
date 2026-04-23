#!/bin/bash
# M2: Baseline Encoder Comparison — train ACT + DP with multiple encoders on libero_10
#
# ACT: ResNet18 frozen/finetuned, ResNet50 frozen, DINOv2-S frozen, SigLIP frozen, scratch
# DP:  ResNet18 default/ImageNet/ResNet50/no-groupnorm
# Each config x 3 seeds x 10 tasks
#
# Usage:
#   bash scripts/cpmae/run_baselines.sh 0 1 2 3   # GPU indices
#
# Estimated: ~150 GPU-hours

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
SEEDS=(42 123 456)
RESULTS_DIR="results/M2_baselines"
REPO_ID="HuggingFaceVLA/libero"
SUITE="libero_10"

mkdir -p "$RESULTS_DIR"

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

# Helper: resolve env task_id
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

# Get task indices for libero_10
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

# Define baseline experiments
# Format: RUN_ID POLICY_TYPE BACKBONE EXTRA_ARGS...
# EXTRA_ARGS are additional lerobot-train flags
EXPERIMENTS=(
  # ACT baselines
  "R100 act resnet18_frozen --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 --policy.freeze_backbone=true"
  "R101 act resnet18_ft --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 --policy.freeze_backbone=false"
  "R102 act resnet50_frozen --policy.vision_backbone=resnet50 --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 --policy.freeze_backbone=true"
  "R103 act dinov2s_frozen --policy.vision_backbone=dinov2 --policy.dinov2_model_name=facebook/dinov2-small --policy.freeze_backbone=true"
  "R104 act siglip_frozen --policy.vision_backbone=siglip --policy.siglip_model_name=google/siglip-base-patch16-224 --policy.freeze_backbone=true"
  "R105 act resnet18_scratch --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=null --policy.freeze_backbone=false"
  # DP baselines
  "R110 dp resnet18_default --policy.vision_backbone=resnet18"
  "R111 dp resnet18_imagenet --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 --policy.use_group_norm=false"
  "R112 dp resnet50_imagenet --policy.vision_backbone=resnet50 --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 --policy.use_group_norm=false"
  "R113 dp resnet18_no_gn --policy.vision_backbone=resnet18 --policy.use_group_norm=false"
)

# Build full job list: (experiment, task, seed)
jobs=()
for exp_line in "${EXPERIMENTS[@]}"; do
  read -r run_id policy_shortname backbone_name extra <<< "$exp_line"
  for seed in "${SEEDS[@]}"; do
    for task_idx in $TASK_INDICES; do
      jobs+=("$run_id|$policy_shortname|$backbone_name|$seed|$task_idx|$extra")
    done
  done
done

echo "Total jobs: ${#jobs[@]} (${#EXPERIMENTS[@]} configs x ${#SEEDS[@]} seeds x $(echo $TASK_INDICES | wc -w) tasks)"
echo "Running on $NUM_GPUS GPUs"
echo ""

run_one_job() {
  local job_str=$1 gpu=$2
  IFS='|' read -r run_id policy_short backbone_name seed task_idx extra <<< "$job_str"

  local episodes=$(resolve_episodes "$task_idx")
  local env_task_id=$(resolve_env_task "$task_idx")
  local run_dir="$RESULTS_DIR/${run_id}_${backbone_name}/seed${seed}/task_${task_idx}"

  if [ -d "$run_dir/checkpoints/last/pretrained_model" ]; then
    echo "[GPU $gpu] $run_id $backbone_name seed=$seed task=$task_idx — completed, skipping"
    return 0
  fi

  echo "[GPU $gpu] $run_id $backbone_name seed=$seed task=$task_idx"

  local policy_type bs lr
  if [ "$policy_short" = "act" ]; then
    policy_type="act"
    bs=$ACT_BS
    lr=$ACT_LR
  else
    policy_type="diffusion"
    bs=$DP_BS
    lr=$DP_LR
  fi

  local cmd_args=(
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
    --seed=$seed
    --output_dir="$run_dir"
    --job_name="${run_id}_${backbone_name}_s${seed}_t${task_idx}"
    --wandb.enable=true
    --wandb.project=cpmae_baselines
    --policy.push_to_hub=false
  )

  # Add policy-specific lr args
  if [ "$policy_type" = "act" ]; then
    cmd_args+=(--policy.optimizer_lr=$lr --policy.optimizer_lr_backbone=$lr)
  else
    cmd_args+=(--policy.optimizer_lr=$lr)
  fi

  # Add extra backbone-specific args
  for arg in $extra; do
    cmd_args+=("$arg")
  done

  CUDA_VISIBLE_DEVICES=$gpu lerobot-train "${cmd_args[@]}"
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

    gpu=${GPUS[$g]}
    run_one_job "${jobs[$idx]}" "$gpu" &
    pids+=($!)
    job_ids+=("${jobs[$idx]}")
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
echo "M2 Baseline Comparison Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
if [ $FAILED_JOBS -gt 0 ]; then
  echo "WARNING: $FAILED_JOBS job(s) failed — check logs above"
fi
echo ""
echo "Collect results:"
echo "  python scripts/cpmae/collect_results.py --input_dir=$RESULTS_DIR --milestone=M2"

exit $( [ $FAILED_JOBS -gt 0 ] && echo 1 || echo 0 )
