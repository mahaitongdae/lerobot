#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <task_index> [batch_size] [learning_rate] [backbone_lr] [freeze_backbone]}
BATCH_SIZE=${2:-8}
LR=${3:-1e-5}
BACKBONE_LR=${4:-1e-6}
FREEZE_BACKBONE=${5:-false}

EPISODES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
task_name = meta.tasks[meta.tasks['task_index'] == $TASK_INDEX].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
")

echo "Training ACT+SigLIP task=$TASK_INDEX batch_size=$BATCH_SIZE lr=$LR backbone_lr=$BACKBONE_LR freeze=$FREEZE_BACKBONE episodes=$EPISODES"

lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="$EPISODES" \
  --policy.type=act \
  --policy.vision_backbone=siglip \
  --policy.siglip_model_name=google/siglip-base-patch16-224 \
  --policy.freeze_backbone=$FREEZE_BACKBONE \
  --batch_size=$BATCH_SIZE \
  --policy.optimizer_lr=$LR \
  --policy.optimizer_lr_backbone=$BACKBONE_LR \
  --output_dir=outputs/train/act_siglip_libero_task${TASK_INDEX}_bs${BATCH_SIZE}_lr${LR}_bblr${BACKBONE_LR}_freeze${FREEZE_BACKBONE} \
  --job_name=act_siglip_libero_task${TASK_INDEX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
