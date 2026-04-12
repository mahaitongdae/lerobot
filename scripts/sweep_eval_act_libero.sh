#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <task_index>}

BATCH_SIZES=(16)
LRS=(5e-5)

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

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

echo "Evaluating sweep results for task $TASK_INDEX (suite=$SUITE, env_task_id=$ENV_TASK_ID)"

# Find all sweep output dirs matching the pattern
jobs=()
for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LRS[@]}"; do
    # Match the output dir pattern from train script (timestamp suffix)
    dir=$(ls -d outputs/train/act_libero_task${TASK_INDEX}_bs${bs}_lr${lr}_* 2>/dev/null | sort | tail -1)
    if [ -z "$dir" ]; then
      echo "SKIP: no training output found for bs=$bs lr=$lr"
      continue
    fi
    checkpoint="$dir/checkpoints/last/pretrained_model"
    if [ ! -d "$checkpoint" ]; then
      echo "SKIP: no checkpoint at $checkpoint"
      continue
    fi
    jobs+=("$bs $lr $checkpoint")
  done
done

if [ ${#jobs[@]} -eq 0 ]; then
    echo "No sweep results found to evaluate."
    exit 1
fi

echo "Found ${#jobs[@]} configs to evaluate"

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break
    read -r bs lr checkpoint <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}
    eval_dir="outputs/eval/act_libero_task${TASK_INDEX}_bs${bs}_lr${lr}"
    echo "Evaluating bs=$bs lr=$lr on GPU $gpu -> $eval_dir"
    CUDA_VISIBLE_DEVICES=$gpu lerobot-eval \
      --policy.path="$checkpoint" \
      --env.type=libero \
      --env.task=$SUITE \
      --env.task_ids="[$ENV_TASK_ID]" \
      --eval.batch_size=16 \
      --eval.n_episodes=100 \
      --output_dir="$eval_dir" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  i=$((i + NUM_GPUS))
done

echo "All sweep evals complete."
