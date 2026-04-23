#!/bin/bash
# Multi-task dry-run: train ACT and DP on all LIBERO-10 tasks jointly (1000 steps)
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/run_multitask_dryrun.sh        # single GPU
#   bash scripts/cpmae/run_multitask_dryrun.sh 0 1    # 2 GPUs (ACT on 0, DP on 1)

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}

STEPS=1000
EVAL_FREQ=500
SAVE_FREQ=1000
N_EVAL_EPISODES=10
EVAL_BATCH=5
BATCH_SIZE=32
LR=1e-4
SEED=42
RESULTS_DIR="results/multitask_dryrun"
REPO_ID="HuggingFaceVLA/libero"
SUITE="libero_10"

MAPPING_JSON="scripts/cpmae/task_mapping.json"

if [ ! -f "$MAPPING_JSON" ]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

# Read from pre-built JSON
NUM_TASKS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(m['suites']['$SUITE']['num_tasks'])")
ALL_EPISODES=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['all_episodes']) + ']')")
ENV_TASK_IDS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['env_task_ids']) + ']')")

mkdir -p "$RESULTS_DIR"

# Trap Ctrl+C
cleanup() {
  echo ""; echo "Caught interrupt — killing background jobs..."
  kill $(jobs -p) 2>/dev/null; wait 2>/dev/null
  echo "All jobs stopped."; exit 1
}
trap cleanup SIGINT SIGTERM

echo "=== Multi-task dry-run: $SUITE ($NUM_TASKS tasks, $STEPS steps) ==="
echo "  Episodes: $(echo "$ALL_EPISODES" | tr -cd ',' | wc -c | xargs) + 1"
echo "  Env task IDs: $ENV_TASK_IDS"
echo ""

# ---- Launch training ----
jobs_list=("act" "dp")
pids=()
job_names=()

for j in "${!jobs_list[@]}"; do
  policy_type="${jobs_list[$j]}"
  gpu_idx=$((j % NUM_GPUS))
  gpu=${GPUS[$gpu_idx]}

  if [ "$policy_type" = "act" ]; then
    RUN_NAME="MT_act_bs${BATCH_SIZE}_lr${LR}"
    RUN_DIR="$RESULTS_DIR/$RUN_NAME"

    echo "[GPU $gpu] $RUN_NAME (ACT, multi-task, $NUM_TASKS tasks)"
    CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
      --dataset.repo_id=$REPO_ID \
      --dataset.episodes="$ALL_EPISODES" \
      --policy.type=act \
      --policy.vision_backbone=resnet18 \
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
      --policy.freeze_backbone=true \
      --policy.num_tasks=$NUM_TASKS \
      --policy.task_embed_dim=64 \
      --env.type=libero \
      --env.task=$SUITE \
      --env.task_ids="$ENV_TASK_IDS" \
      --batch_size=$BATCH_SIZE \
      --steps=$STEPS \
      --eval_freq=$EVAL_FREQ \
      --save_freq=$SAVE_FREQ \
      --eval.n_episodes=$N_EVAL_EPISODES \
      --eval.batch_size=$EVAL_BATCH \
      --seed=$SEED \
      --policy.optimizer_lr=$LR \
      --policy.optimizer_lr_backbone=$LR \
      --output_dir="$RUN_DIR" \
      --job_name="$RUN_NAME" \
      --wandb.enable=false \
      --policy.push_to_hub=false &
    pids+=($!)
    job_names+=("$RUN_NAME")

  elif [ "$policy_type" = "dp" ]; then
    RUN_NAME="MT_dp_bs${BATCH_SIZE}_lr${LR}"
    RUN_DIR="$RESULTS_DIR/$RUN_NAME"

    echo "[GPU $gpu] $RUN_NAME (DP, multi-task, $NUM_TASKS tasks)"
    CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
      --dataset.repo_id=$REPO_ID \
      --dataset.episodes="$ALL_EPISODES" \
      --policy.type=diffusion \
      --policy.vision_backbone=resnet18 \
      --policy.num_tasks=$NUM_TASKS \
      --policy.task_embed_dim=64 \
      --env.type=libero \
      --env.task=$SUITE \
      --env.task_ids="$ENV_TASK_IDS" \
      --batch_size=$BATCH_SIZE \
      --steps=$STEPS \
      --eval_freq=$EVAL_FREQ \
      --save_freq=$SAVE_FREQ \
      --eval.n_episodes=$N_EVAL_EPISODES \
      --eval.batch_size=$EVAL_BATCH \
      --seed=$SEED \
      --policy.optimizer_lr=$LR \
      --output_dir="$RUN_DIR" \
      --job_name="$RUN_NAME" \
      --wandb.enable=false \
      --policy.push_to_hub=false &
    pids+=($!)
    job_names+=("$RUN_NAME")
  fi

  # If only 1 GPU, wait for current job before launching next
  if [ $NUM_GPUS -eq 1 ] && [ $j -lt $((${#jobs_list[@]} - 1)) ]; then
    echo "  (single GPU: waiting for ${job_names[-1]} to finish...)"
    wait "${pids[-1]}" || echo "  WARNING: ${job_names[-1]} failed"
  fi
done

# Wait for remaining jobs
FAILED=0
for k in "${!pids[@]}"; do
  if ! wait "${pids[$k]}"; then
    echo "FAILED: ${job_names[$k]}"
    FAILED=$((FAILED + 1))
  else
    echo "DONE: ${job_names[$k]}"
  fi
done

echo ""
echo "========================================="
echo "Multi-task dry-run complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
[ $FAILED -gt 0 ] && echo "WARNING: $FAILED job(s) failed"
echo ""

exit $( [ $FAILED -gt 0 ] && echo 1 || echo 0 )
