#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <dataset_task_index>}

# Resolve dataset task_index -> (suite_name, env_task_id)
read -r SUITE ENV_TASK_ID <<< $(python3 -c "
import sys, io, os
os.environ['LIBERO_QUIET'] = '1'
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark

meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
ds_tasks = {int(row['task_index']): name for name, row in meta.tasks.iterrows()}
task_name = ds_tasks[$TASK_INDEX].strip().lower()

for suite_name in ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i in range(len(suite.tasks)):
        if suite.get_task(i).language.strip().lower() == task_name:
            sys.stdout = _real_stdout
            print(suite_name, i)
            exit()
sys.stdout = _real_stdout
print('NOT_FOUND -1')
")

if [ "$SUITE" = "NOT_FOUND" ]; then
    echo "ERROR: dataset task_index=$TASK_INDEX not found in any LIBERO suite"
    exit 1
fi

echo "Dataset task_index=$TASK_INDEX -> env suite=$SUITE, env task_id=$ENV_TASK_ID"

lerobot-eval \
  --policy.path=outputs/train/act_libero_task${TASK_INDEX}_20260308_144828/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=$SUITE \
  --env.task_ids="[$ENV_TASK_ID]" \
  --eval.batch_size=8 \
  --eval.n_episodes=8
