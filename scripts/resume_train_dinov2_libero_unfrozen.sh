#!/bin/bash
# Resume training: ACT + DINOv2-base (unfrozen encoder) from checkpoint 060000
# for tasks 0-3 on 4 GPUs in parallel.
# Press Ctrl+C to stop all running jobs.

set -euo pipefail

CHECKPOINT_BASE="/mnt/shared/haitongma/wdir/lerobot/outputs/train"
CHECKPOINT_STEP="060000"

TASK_DIRS=(
  "act_dinov2-base_libero_task0_bs4x4_lr1e-5_unfrozen_20260310_235754"
  # "act_dinov2-base_libero_task1_bs4x4_lr1e-5_unfrozen_20260310_235757"
  # "act_dinov2-base_libero_task2_bs4x4_lr1e-5_unfrozen_20260310_235759"
  # "act_dinov2-base_libero_task3_bs4x4_lr1e-5_unfrozen_20260310_235801"
)

GPUS=(0)

# Trap Ctrl+C to kill all background jobs
cleanup() {
  echo ""
  echo "Caught interrupt — killing all background jobs..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "All jobs stopped."
  exit 1
}
trap cleanup SIGINT SIGTERM

pids=()
for i in "${!TASK_DIRS[@]}"; do
  gpu=${GPUS[$i]}
  task_dir="${TASK_DIRS[$i]}"
  config_path="${CHECKPOINT_BASE}/${task_dir}/checkpoints/${CHECKPOINT_STEP}/pretrained_model/train_config.json"

  echo "[GPU $gpu] Resuming ${task_dir} from checkpoint ${CHECKPOINT_STEP}"

  CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
    --config_path="$config_path" \
    --resume=true &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo ""
echo "All resume training jobs complete."
