#!/bin/bash
# Sweep training: ACT + DINOv2-base (unfrozen) with 2 backbone LRs
# Head LR fixed at 1e-5, backbone LR swept over {1e-6, 5e-7}
# 4 tasks × 2 LRs = 8 jobs, run 4 at a time on 4 GPUs.

set -euo pipefail

LR=1e-5
BACKBONE_LRS=(1e-6 5e-7)
FREEZE=false

TASK_INDICES=(0 1 2 3)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

BB_TYPE=dinov2
MODEL_NAME=facebook/dinov2-base
MODEL_SHORT=dinov2-base
BS=4
GA=4

# Hardcoded episodes per task (from HuggingFaceVLA/libero)
declare -A EPISODES
EPISODES[0]="[0,18,22,33,58,85,88,105,107,114,121,125,129,157,167,170,190,207,211,231,233,235,236,247,249,257,264,267,295,301,307,309,315,323,343,346,362,367]"
EPISODES[1]="[1,4,5,11,19,21,37,52,62,68,108,110,113,130,138,146,152,168,176,197,206,210,212,216,220,224,242,248,252,271,302,322,325,355,356,369]"
EPISODES[2]="[2,3,34,35,44,59,80,84,99,123,126,139,140,142,145,164,173,178,186,193,217,226,230,245,269,276,282,285,312,324,339,342,357,365]"
EPISODES[3]="[6,38,40,45,48,49,50,66,72,87,93,95,134,150,153,162,182,184,185,202,203,218,225,239,240,243,253,259,263,272,277,278,287,292,303,321,345,351,354,358,361]"

cleanup() {
  echo ""
  echo "Caught interrupt — killing all background jobs..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "All jobs stopped."
  exit 1
}
trap cleanup SIGINT SIGTERM

# Collect all jobs: "backbone_lr task_idx"
jobs=()
for bblr in "${BACKBONE_LRS[@]}"; do
  for task_idx in "${TASK_INDICES[@]}"; do
    jobs+=("$bblr $task_idx")
  done
done

echo "Total jobs: ${#jobs[@]} (${#BACKBONE_LRS[@]} backbone LRs × ${#TASK_INDICES[@]} tasks)"
echo "Head LR=$LR, backbone LRs=${BACKBONE_LRS[*]}"
echo ""

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break

    read -r bblr task_idx <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}

    OUT_DIR="/mnt/shared/haitongma/wdir/lerobot/outputs/train/act_${MODEL_SHORT}_libero_task${task_idx}_bs${BS}x${GA}_lr${LR}_bblr${bblr}_unfrozen"
    JOB_NAME="act_${MODEL_SHORT}_task${task_idx}_bblr${bblr}"

    echo "[GPU $gpu] task=$task_idx backbone_lr=$bblr -> $OUT_DIR"

    CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
      --dataset.repo_id=HuggingFaceVLA/libero \
      --dataset.episodes="${EPISODES[$task_idx]}" \
      --policy.type=act \
      --policy.vision_backbone=$BB_TYPE \
      --policy.${BB_TYPE}_model_name=$MODEL_NAME \
      --policy.freeze_backbone=$FREEZE \
      --batch_size=$BS \
      --gradient_accumulation_steps=$GA \
      --policy.optimizer_lr=$LR \
      --policy.optimizer_lr_backbone=$bblr \
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
