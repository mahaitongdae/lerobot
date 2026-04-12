#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <task_index> [batch_size] [learning_rate]}
BATCH_SIZE=${2:-8}
LR=${3:-1e-5}

EPISODES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
task_name = meta.tasks[meta.tasks['task_index'] == $TASK_INDEX].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
")

echo "Training task=$TASK_INDEX batch_size=$BATCH_SIZE lr=$LR episodes=$EPISODES"

lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="$EPISODES" \
  --policy.type=act \
  --batch_size=$BATCH_SIZE \
  --policy.optimizer_lr=$LR \
  --policy.optimizer_lr_backbone=$LR \
  --output_dir=outputs/train/act_libero_task${TASK_INDEX}_bs${BATCH_SIZE}_lr${LR}_$(date +%Y%m%d_%H%M%S) \
  --job_name=act_libero_task${TASK_INDEX}_bs${BATCH_SIZE}_lr${LR} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
