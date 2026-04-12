#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <task_index> [batch_size] [learning_rate] [backbone_lr] [freeze_backbone] [dinov2_model]}
BATCH_SIZE=${2:-8}
LR=${3:-1e-5}
BACKBONE_LR=${4:-1e-6}
FREEZE_BACKBONE=${5:-false}
DINOV2_MODEL=${6:-facebook/dinov2-small}

EPISODES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
task_name = meta.tasks[meta.tasks['task_index'] == $TASK_INDEX].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
")

MODEL_SHORT=$(echo "$DINOV2_MODEL" | sed 's|.*/||')
echo "Training ACT+DINOv2 (${MODEL_SHORT}) task=$TASK_INDEX batch_size=$BATCH_SIZE lr=$LR backbone_lr=$BACKBONE_LR freeze=$FREEZE_BACKBONE episodes=$EPISODES"

lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="$EPISODES" \
  --policy.type=act \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=$DINOV2_MODEL \
  --policy.freeze_backbone=$FREEZE_BACKBONE \
  --batch_size=$BATCH_SIZE \
  --policy.optimizer_lr=$LR \
  --policy.optimizer_lr_backbone=$BACKBONE_LR \
  --output_dir=outputs/train/act_${MODEL_SHORT}_libero_task${TASK_INDEX}_bs${BATCH_SIZE}_lr${LR}_bblr${BACKBONE_LR}_freeze${FREEZE_BACKBONE}_$(date +%Y%m%d_%H%M%S) \
  --job_name=act_${MODEL_SHORT}_libero_task${TASK_INDEX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
