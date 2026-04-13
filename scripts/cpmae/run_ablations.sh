#!/bin/bash
# M4: Ablation Studies — CP-MAE hyperparameter sensitivity
#
# M4a: Mask ratio sensitivity
# M4b: Loss weight sensitivity
# M4c: Contact window size
# M4d: Analysis (CKA, attention maps, reconstruction MSE)
#
# Depends on M3 (CP-MAE checkpoint + baseline results)
#
# Usage:
#   bash scripts/cpmae/run_ablations.sh 0 1 2 3   # GPU indices
#
# Estimated: ~150 GPU-hours

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}
GPU0=${GPUS[0]}

ACT_BS=8
ACT_LR=1e-5
PRETRAIN_EPOCHS=400
PRETRAIN_BS=256
DOWNSTREAM_STEPS=100000
EVAL_FREQ=10000
SAVE_FREQ=50000
N_EVAL_EPISODES=20
EVAL_BATCH=20
SEED=42
RESULTS_DIR="results/M4_ablations"
REPO_ID="HuggingFaceVLA/libero"
CONTACT_LABELS_DIR="results/contact_labels"
SUITE="libero_10"

mkdir -p "$RESULTS_DIR"

cleanup() {
  echo ""; echo "Caught interrupt — killing background jobs..."
  kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; exit 1
}
trap cleanup SIGINT SIGTERM

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

# =========================================================================
# M4a: Mask Ratio Sensitivity
# =========================================================================

echo ""
echo "========================================="
echo "M4a: Mask Ratio Sensitivity"
echo "========================================="

# Pretrain with different contact mask ratios, then evaluate downstream
MASK_ABLATIONS=(
  "R300 0.80 0.75 2.0"
  "R301 0.85 0.75 2.0"
  "R302 0.95 0.75 2.0"
)

for abl in "${MASK_ABLATIONS[@]}"; do
  read -r run_id cmr tmr lw <<< "$abl"
  abl_dir="$RESULTS_DIR/${run_id}_cmr${cmr}_tmr${tmr}"

  if [ ! -f "$abl_dir/encoder_final.pt" ]; then
    echo "Pretraining $run_id (contact_mask=$cmr, transit_mask=$tmr, loss_weight=$lw)..."
    CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/cpmae/pretrain_mae.py \
      --mode=cpmae \
      --output_dir="$abl_dir" \
      --epochs=$PRETRAIN_EPOCHS \
      --batch_size=$PRETRAIN_BS \
      --contact_mask_ratio=$cmr \
      --transit_mask_ratio=$tmr \
      --contact_loss_weight=$lw \
      --contact_labels_dir="$CONTACT_LABELS_DIR" \
      --gpu=0
  else
    echo "$run_id pretrained checkpoint exists"
  fi
done

# =========================================================================
# M4b: Loss Weight Sensitivity
# =========================================================================

echo ""
echo "========================================="
echo "M4b: Loss Weight Sensitivity"
echo "========================================="

WEIGHT_ABLATIONS=(
  "R310 0.90 0.75 1.0"
  "R311 0.90 0.75 1.5"
  "R312 0.90 0.75 4.0"
)

for abl in "${WEIGHT_ABLATIONS[@]}"; do
  read -r run_id cmr tmr lw <<< "$abl"
  abl_dir="$RESULTS_DIR/${run_id}_lw${lw}"

  if [ ! -f "$abl_dir/encoder_final.pt" ]; then
    echo "Pretraining $run_id (loss_weight=$lw)..."
    CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/cpmae/pretrain_mae.py \
      --mode=cpmae \
      --output_dir="$abl_dir" \
      --epochs=$PRETRAIN_EPOCHS \
      --batch_size=$PRETRAIN_BS \
      --contact_mask_ratio=$cmr \
      --transit_mask_ratio=$tmr \
      --contact_loss_weight=$lw \
      --contact_labels_dir="$CONTACT_LABELS_DIR" \
      --gpu=0
  else
    echo "$run_id pretrained checkpoint exists"
  fi
done

# =========================================================================
# M4c: Contact Window Size
# =========================================================================

echo ""
echo "========================================="
echo "M4c: Contact Window Size"
echo "========================================="

WINDOW_ABLATIONS=(
  "R320 1"
  "R321 3"
  "R322 10"
)

for abl in "${WINDOW_ABLATIONS[@]}"; do
  read -r run_id window <<< "$abl"
  label_dir="$RESULTS_DIR/contact_labels_w${window}"
  abl_dir="$RESULTS_DIR/${run_id}_window${window}"

  # Regenerate contact labels with different window
  if [ ! -d "$label_dir" ]; then
    echo "Generating contact labels with window=$window..."
    python3 scripts/cpmae/contact_detector.py --all_tasks --window=$window --output_dir="$label_dir"
  fi

  if [ ! -f "$abl_dir/encoder_final.pt" ]; then
    echo "Pretraining $run_id (window=$window)..."
    CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/cpmae/pretrain_mae.py \
      --mode=cpmae \
      --output_dir="$abl_dir" \
      --epochs=$PRETRAIN_EPOCHS \
      --batch_size=$PRETRAIN_BS \
      --contact_mask_ratio=0.90 \
      --transit_mask_ratio=0.75 \
      --contact_loss_weight=2.0 \
      --contact_labels_dir="$label_dir" \
      --gpu=0
  else
    echo "$run_id pretrained checkpoint exists"
  fi
done

# =========================================================================
# Downstream evaluation for all ablation encoders
# =========================================================================

echo ""
echo "========================================="
echo "Downstream Training for All Ablations"
echo "========================================="

# Collect all ablation encoder checkpoints
ABL_ENCODERS=()
for abl in "${MASK_ABLATIONS[@]}"; do
  read -r run_id cmr tmr lw <<< "$abl"
  ABL_ENCODERS+=("$run_id|$RESULTS_DIR/${run_id}_cmr${cmr}_tmr${tmr}/encoder_final.pt")
done
for abl in "${WEIGHT_ABLATIONS[@]}"; do
  read -r run_id cmr tmr lw <<< "$abl"
  ABL_ENCODERS+=("$run_id|$RESULTS_DIR/${run_id}_lw${lw}/encoder_final.pt")
done
for abl in "${WINDOW_ABLATIONS[@]}"; do
  read -r run_id window <<< "$abl"
  ABL_ENCODERS+=("$run_id|$RESULTS_DIR/${run_id}_window${window}/encoder_final.pt")
done

# Build job list for downstream training
abl_jobs=()
for enc_entry in "${ABL_ENCODERS[@]}"; do
  IFS='|' read -r run_id ckpt_path <<< "$enc_entry"
  for task_idx in $TASK_INDICES; do
    abl_jobs+=("$run_id|$ckpt_path|$task_idx")
  done
done

echo "Ablation downstream jobs: ${#abl_jobs[@]}"

run_abl_downstream() {
  local job_str=$1 gpu=$2
  IFS='|' read -r run_id ckpt_path task_idx <<< "$job_str"

  local episodes=$(resolve_episodes "$task_idx")
  local env_task_id=$(resolve_env_task "$task_idx")
  local run_dir="$RESULTS_DIR/${run_id}_downstream/task_${task_idx}"

  if [ -d "$run_dir/checkpoints/last/pretrained_model" ]; then
    echo "[GPU $gpu] $run_id task=$task_idx — completed, skipping"
    return 0
  fi

  if [ ! -f "$ckpt_path" ]; then
    echo "[GPU $gpu] $run_id task=$task_idx — checkpoint not found: $ckpt_path, skipping"
    return 0
  fi

  echo "[GPU $gpu] $run_id task=$task_idx"

  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/cpmae/train_with_cpmae.py \
    --cpmae_checkpoint="$ckpt_path" \
    --cpmae_freeze=true \
    -- \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$episodes" \
    --policy.type=act \
    --policy.vision_backbone=cpmae \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="[$env_task_id]" \
    --batch_size=$ACT_BS \
    --steps=$DOWNSTREAM_STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$SEED \
    --policy.optimizer_lr=$ACT_LR \
    --policy.optimizer_lr_backbone=$ACT_LR \
    --output_dir="$run_dir" \
    --job_name="${run_id}_t${task_idx}" \
    --wandb.enable=true \
    --wandb.project=cpmae_ablations \
    --policy.push_to_hub=false
}

i=0
FAILED_JOBS=0
while [ $i -lt ${#abl_jobs[@]} ]; do
  pids=()
  job_ids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#abl_jobs[@]} ] && break
    gpu=${GPUS[$g]}
    run_abl_downstream "${abl_jobs[$idx]}" "$gpu" &
    pids+=($!)
    job_ids+=("${abl_jobs[$idx]}")
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
echo "M4 Ablations Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
if [ $FAILED_JOBS -gt 0 ]; then
  echo "WARNING: $FAILED_JOBS job(s) failed — check logs above"
fi
echo ""
echo "Collect results:"
echo "  python scripts/cpmae/collect_results.py --input_dir=$RESULTS_DIR --milestone=M4"

exit $( [ $FAILED_JOBS -gt 0 ] && echo 1 || echo 0 )
