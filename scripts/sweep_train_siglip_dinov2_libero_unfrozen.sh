#!/bin/bash
# Sweep training: ACT + SigLIP-base and ACT + DINOv2-base (unfrozen encoder, lr=1e-5)
# across 4 tasks on 4 GPUs.
# Press Ctrl+C to stop all running jobs.

set -euo pipefail

LR=1e-5
BACKBONE_LR=1e-5
FREEZE=false

TASK_INDICES=(0 1 2 3)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

BACKBONES=(
  "siglip google/siglip-base-patch16-224 16 1"
  "dinov2 facebook/dinov2-base 4 4"
)

# Collect all (backbone_type, model_name, task_index, batch_size, grad_accum) combos
jobs=()
for entry in "${BACKBONES[@]}"; do
  read -r bb_type model_name bs ga <<< "$entry"
  for task in "${TASK_INDICES[@]}"; do
    jobs+=("$bb_type $model_name $bs $ga $task")
  done
done

echo "Total jobs: ${#jobs[@]} (${#BACKBONES[@]} backbones × ${#TASK_INDICES[@]} tasks)"
echo "lr=$LR backbone_lr=$BACKBONE_LR freeze=$FREEZE"
echo ""

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

resolve_episodes() {
  local task_idx=$1
  python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
task_name = meta.tasks[meta.tasks['task_index'] == $task_idx].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
"
}

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break

    read -r bb_type model_name bs ga task_idx <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}
    model_short=$(echo "$model_name" | sed 's|.*/||')

    EPISODES=$(resolve_episodes "$task_idx")

    OUT_DIR="outputs/train/act_${model_short}_libero_task${task_idx}_bs${bs}x${ga}_lr${LR}_unfrozen_$(date +%Y%m%d_%H%M%S)"
    JOB_NAME="act_${model_short}_task${task_idx}_unfrozen"

    echo "[GPU $gpu] $bb_type ($model_short) task=$task_idx bs=${bs}x${ga} episodes=$EPISODES -> $OUT_DIR"

    CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
      --dataset.repo_id=HuggingFaceVLA/libero \
      --dataset.episodes="$EPISODES" \
      --policy.type=act \
      --policy.vision_backbone=$bb_type \
      --policy.${bb_type}_model_name=$model_name \
      --policy.freeze_backbone=$FREEZE \
      --batch_size=$bs \
      --gradient_accumulation_steps=$ga \
      --policy.optimizer_lr=$LR \
      --policy.optimizer_lr_backbone=$BACKBONE_LR \
      --output_dir="$OUT_DIR" \
      --job_name="$JOB_NAME" \
      --policy.device=cuda \
      --wandb.enable=true \
      --policy.push_to_hub=false &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  i=$((i + NUM_GPUS))
done

echo ""
echo "All sweep training jobs complete."
