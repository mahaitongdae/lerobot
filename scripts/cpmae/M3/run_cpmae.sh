#!/bin/bash
# M3: CP-MAE Pretraining + Downstream Policy Training
#
# Phase 1 (M3a): Pretrain CP-MAE and Uniform MAE encoders on LIBERO images
# Phase 2 (M3b): Train ACT policies with pretrained encoders on libero_10
# Phase 3 (M3c): Sample efficiency study (10/25/50% data fractions)
#
# Depends on:
#   - Contact labels: results/contact_labels/ (from contact_detector.py)
#   - M2 results for comparison
#
# Usage:
#   bash scripts/cpmae/run_cpmae.sh 0 1 2 3   # GPU indices
#
# Estimated: ~250 GPU-hours

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}
GPU0=${GPUS[0]}

# Best hyperparameters from M0+
ACT_BS=8
ACT_LR=1e-5

STEPS=100000
EVAL_FREQ=10000
SAVE_FREQ=50000
N_EVAL_EPISODES=20
EVAL_BATCH=20
SEEDS=(42 123 456)
RESULTS_DIR="results/M3_cpmae"
REPO_ID="HuggingFaceVLA/libero"
CONTACT_LABELS_DIR="results/contact_labels"
SUITE="libero_10"
MAPPING_JSON="scripts/cpmae/task_mapping.json"

mkdir -p "$RESULTS_DIR"

# Trap Ctrl+C
cleanup() {
  echo ""; echo "Caught interrupt — killing background jobs..."
  kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; exit 1
}
trap cleanup SIGINT SIGTERM

# Ensure task_mapping.json exists
if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

# Resolve multi-task metadata from task_mapping.json
NUM_TASKS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(m['suites']['$SUITE']['num_tasks'])")
ALL_EPISODES=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['all_episodes']) + ']')")
ENV_TASK_IDS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['env_task_ids']) + ']')")
TASK_INDEX_OFFSET=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(min(m['suites']['$SUITE']['dataset_task_indices']))")

echo "Suite $SUITE: $NUM_TASKS tasks, offset=$TASK_INDEX_OFFSET"

resolve_episodes_fraction() {
  # Return a fraction of episodes across all tasks (stratified by task)
  local fraction=$1 seed=$2
  python3 -c "
import json, random
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
m = json.load(open('$MAPPING_JSON'))
suite = m['suites']['$SUITE']
meta = LeRobotDatasetMetadata('$REPO_ID')
sampled = []
for task_idx in suite['dataset_task_indices']:
    task_name = meta.tasks[meta.tasks['task_index'] == task_idx].index[0]
    eps = sorted(ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks'])
    random.seed($seed + task_idx)
    n = max(1, int(len(eps) * $fraction))
    sampled.extend(random.sample(eps, n))
print('[' + ','.join(str(e) for e in sorted(sampled)) + ']')
"
}

# =========================================================================
# Phase 1: Pretrain encoders (M3a)
# =========================================================================

echo ""
echo "========================================="
echo "M3a: CP-MAE Pretraining"
echo "========================================="

# Ensure contact labels exist
if [ ! -d "$CONTACT_LABELS_DIR" ] || [ -z "$(ls -A $CONTACT_LABELS_DIR 2>/dev/null)" ]; then
  echo "Generating contact labels..."
  python3 scripts/cpmae/contact_detector.py --all_tasks --output_dir="$CONTACT_LABELS_DIR"
fi

# R200: CP-MAE
CPMAE_DIR="$RESULTS_DIR/R200_cpmae"
if [ ! -f "$CPMAE_DIR/encoder_final.pt" ]; then
  echo "Training CP-MAE encoder..."
  CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/cpmae/pretrain_mae.py \
    --mode=cpmae \
    --output_dir="$CPMAE_DIR" \
    --epochs=400 \
    --batch_size=256 \
    --lr=1.5e-4 \
    --contact_mask_ratio=0.90 \
    --transit_mask_ratio=0.75 \
    --contact_loss_weight=2.0 \
    --contact_labels_dir="$CONTACT_LABELS_DIR" \
    --gpu=0
  echo "R200 CP-MAE DONE"
else
  echo "R200 CP-MAE already exists"
fi

# R201: Uniform MAE
UMAE_DIR="$RESULTS_DIR/R201_uniform_mae"
if [ ! -f "$UMAE_DIR/encoder_final.pt" ]; then
  echo "Training Uniform MAE encoder..."
  CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/cpmae/pretrain_mae.py \
    --mode=uniform \
    --output_dir="$UMAE_DIR" \
    --epochs=400 \
    --batch_size=256 \
    --lr=1.5e-4 \
    --uniform_mask_ratio=0.75 \
    --gpu=0
  echo "R201 Uniform MAE DONE"
else
  echo "R201 Uniform MAE already exists"
fi

CPMAE_CKPT="$CPMAE_DIR/encoder_final.pt"
UMAE_CKPT="$UMAE_DIR/encoder_final.pt"

# =========================================================================
# Phase 2: Downstream policy training (M3b)
# =========================================================================

echo ""
echo "========================================="
echo "M3b: Downstream ACT Training with CP-MAE"
echo "========================================="

# Define downstream experiments: RUN_ID ENCODER FREEZE CHECKPOINT
DOWNSTREAM_EXPS=(
  "R210 cpmae true $CPMAE_CKPT"
  "R211 cpmae false $CPMAE_CKPT"
  "R212 umae true $UMAE_CKPT"
  "R213 umae false $UMAE_CKPT"
)

# Build job list: (run_id, encoder_name, freeze, ckpt, seed)
ds_jobs=()
for exp in "${DOWNSTREAM_EXPS[@]}"; do
  read -r run_id enc_name freeze ckpt <<< "$exp"
  for seed in "${SEEDS[@]}"; do
    ds_jobs+=("$run_id|$enc_name|$freeze|$ckpt|$seed")
  done
done

echo "Downstream jobs: ${#ds_jobs[@]}"

run_downstream_job() {
  local job_str=$1 gpu=$2
  IFS='|' read -r run_id enc_name freeze ckpt seed <<< "$job_str"

  local freeze_str=$([ "$freeze" = "true" ] && echo "frozen" || echo "ft")
  local run_dir="$RESULTS_DIR/${run_id}_${enc_name}_${freeze_str}/seed${seed}"

  if [ -d "$run_dir/checkpoints/last/pretrained_model" ]; then
    echo "[GPU $gpu] $run_id $enc_name $freeze_str seed=$seed — completed, skipping"
    return 0
  fi

  echo "[GPU $gpu] $run_id $enc_name $freeze_str seed=$seed (multi-task, $NUM_TASKS tasks)"

  CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$ALL_EPISODES" \
    --policy.type=act \
    --policy.vision_backbone=cpmae \
    --policy.cpmae_checkpoint_path="$ckpt" \
    --policy.freeze_backbone=$freeze \
    --policy.num_tasks=$NUM_TASKS \
    --policy.task_embed_dim=64 \
    --policy.task_index_offset=$TASK_INDEX_OFFSET \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="$ENV_TASK_IDS" \
    --batch_size=$ACT_BS \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$seed \
    --policy.optimizer_lr=$ACT_LR \
    --policy.optimizer_lr_backbone=$ACT_LR \
    --output_dir="$run_dir" \
    --job_name="${run_id}_${enc_name}_${freeze_str}_s${seed}" \
    --wandb.enable=true \
    --wandb.project=cpmae_downstream \
    --policy.push_to_hub=false
}

# Dispatch downstream jobs
FAILED_JOBS=0
i=0
while [ $i -lt ${#ds_jobs[@]} ]; do
  pids=()
  job_ids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#ds_jobs[@]} ] && break
    gpu=${GPUS[$g]}
    run_downstream_job "${ds_jobs[$idx]}" "$gpu" &
    pids+=($!)
    job_ids+=("${ds_jobs[$idx]}")
  done
  for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then
      echo "FAILED: ${job_ids[$j]}"
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
  done
  i=$((i + NUM_GPUS))
done

# =========================================================================
# Phase 3: Sample efficiency (M3c)
# =========================================================================

echo ""
echo "========================================="
echo "M3c: Sample Efficiency Study"
echo "========================================="

DATA_FRACTIONS=(0.10 0.25 0.50)
EFFICIENCY_EXPS=(
  "R220 cpmae $CPMAE_CKPT"
  "R221 umae $UMAE_CKPT"
  "R223 imagenet NONE"
)

eff_jobs=()
for exp in "${EFFICIENCY_EXPS[@]}"; do
  read -r run_base enc_name ckpt <<< "$exp"
  for frac in "${DATA_FRACTIONS[@]}"; do
    frac_pct=$(python3 -c "print(int($frac * 100))")
    for seed in "${SEEDS[@]}"; do
      eff_jobs+=("${run_base}_${frac_pct}pct|$enc_name|$ckpt|$frac|$seed")
    done
  done
done

echo "Sample efficiency jobs: ${#eff_jobs[@]}"

run_efficiency_job() {
  local job_str=$1 gpu=$2
  IFS='|' read -r run_id enc_name ckpt frac seed <<< "$job_str"

  local episodes=$(resolve_episodes_fraction "$frac" "$seed")
  local run_dir="$RESULTS_DIR/${run_id}/seed${seed}"

  if [ -d "$run_dir/checkpoints/last/pretrained_model" ]; then
    echo "[GPU $gpu] $run_id seed=$seed — completed, skipping"
    return 0
  fi

  echo "[GPU $gpu] $run_id ($enc_name, ${frac}x data, multi-task) seed=$seed"

  local vision_args=""
  if [ "$enc_name" = "cpmae" ] || [ "$enc_name" = "umae" ]; then
    vision_args="--policy.vision_backbone=cpmae --policy.cpmae_checkpoint_path=$ckpt --policy.freeze_backbone=true"
  else
    vision_args="--policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 --policy.freeze_backbone=true"
  fi

  CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --dataset.episodes="$episodes" \
    --policy.type=act \
    $vision_args \
    --policy.num_tasks=$NUM_TASKS \
    --policy.task_embed_dim=64 \
    --policy.task_index_offset=$TASK_INDEX_OFFSET \
    --env.type=libero \
    --env.task=$SUITE \
    --env.task_ids="$ENV_TASK_IDS" \
    --batch_size=$ACT_BS \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH \
    --seed=$seed \
    --policy.optimizer_lr=$ACT_LR \
    --policy.optimizer_lr_backbone=$ACT_LR \
    --output_dir="$run_dir" \
    --job_name="${run_id}_s${seed}" \
    --wandb.enable=true \
    --wandb.project=cpmae_efficiency \
    --policy.push_to_hub=false
}

i=0
while [ $i -lt ${#eff_jobs[@]} ]; do
  pids=()
  job_ids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#eff_jobs[@]} ] && break
    gpu=${GPUS[$g]}
    run_efficiency_job "${eff_jobs[$idx]}" "$gpu" &
    pids+=($!)
    job_ids+=("${eff_jobs[$idx]}")
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
echo "M3 Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
if [ $FAILED_JOBS -gt 0 ]; then
  echo "WARNING: $FAILED_JOBS job(s) failed — check logs above"
fi
echo ""
echo "Collect results:"
echo "  python scripts/cpmae/collect_results.py --input_dir=$RESULTS_DIR --milestone=M3"

exit $( [ $FAILED_JOBS -gt 0 ] && echo 1 || echo 0 )
