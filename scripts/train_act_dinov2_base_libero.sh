#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <task_index> [batch_size] [learning_rate] [backbone_lr] [freeze_backbone] [grad_accum]}
BATCH_SIZE=${2:-8}
LR=${3:-1e-5}
BACKBONE_LR=${4:-1e-6}
FREEZE_BACKBONE=${5:-false}
GRAD_ACCUM=${6:-2}

EPISODES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
task_name = meta.tasks[meta.tasks['task_index'] == $TASK_INDEX].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
")

echo "Training ACT+DINOv2-base task=$TASK_INDEX batch_size=$BATCH_SIZE lr=$LR backbone_lr=$BACKBONE_LR freeze=$FREEZE_BACKBONE episodes=$EPISODES"

lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="$EPISODES" \
  --policy.type=act \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=facebook/dinov2-base \
  --policy.freeze_backbone=$FREEZE_BACKBONE \
  --batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACCUM \
  --policy.optimizer_lr=$LR \
  --policy.optimizer_lr_backbone=$BACKBONE_LR \
  --output_dir=outputs/train/act_dinov2-base_libero_task${TASK_INDEX}_bs${BATCH_SIZE}_lr${LR}_bblr${BACKBONE_LR}_freeze${FREEZE_BACKBONE}_$(date +%Y%m%d_%H%M%S) \
  --job_name=act_dinov2-base_libero_task${TASK_INDEX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
