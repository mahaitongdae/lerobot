#!/bin/bash
# Sweep evaluation: ACT + SigLIP-base and ACT + DINOv2-base across 4 tasks on 4 GPUs.
# Finds the latest training checkpoint for each (backbone, task) combo and evaluates it.

set -euo pipefail

TASK_INDICES=(1 2 3)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

MODELS=(
  "siglip-base-patch16-224"
  "dinov2-base"
)

# Match the output dir pattern from the training sweep script
TRAIN_DIR_PATTERN="outputs/train"

cleanup() {
  echo ""
  echo "Caught interrupt — killing all background jobs..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "All jobs stopped."
  exit 1
}
trap cleanup SIGINT SIGTERM

# Resolve dataset task_index -> (suite_name, env_task_id)
resolve_libero_task() {
  local task_idx=$1
  python3 -c "
import sys, io, os
os.environ['LIBERO_QUIET'] = '1'
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark

meta = LeRobotDatasetMetadata('HuggingFaceVLA/libero')
ds_tasks = {int(row['task_index']): name for name, row in meta.tasks.iterrows()}
task_name = ds_tasks[$task_idx].strip().lower()

for suite_name in ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i in range(len(suite.tasks)):
        if suite.get_task(i).language.strip().lower() == task_name:
            sys.stdout = _real_stdout
            print(suite_name, i)
            exit()
sys.stdout = _real_stdout
print('NOT_FOUND -1')
"
}

# Collect all (model_short, task_index, checkpoint, suite, env_task_id) jobs
jobs=()
for model_short in "${MODELS[@]}"; do
  for task_idx in "${TASK_INDICES[@]}"; do
    dir=$(ls -d ${TRAIN_DIR_PATTERN}/act_${model_short}_libero_task${task_idx}_* 2>/dev/null | sort | tail -1)
    if [ -z "$dir" ]; then
      echo "SKIP: no training output for model=$model_short task=$task_idx"
      continue
    fi
    checkpoint="$dir/checkpoints/last/pretrained_model"
    if [ ! -d "$checkpoint" ]; then
      echo "SKIP: no checkpoint at $checkpoint"
      continue
    fi

    read -r suite env_task_id <<< $(resolve_libero_task "$task_idx")
    if [ "$suite" = "NOT_FOUND" ]; then
      echo "SKIP: task_index=$task_idx not found in any LIBERO suite"
      continue
    fi

    jobs+=("$model_short $task_idx $suite $env_task_id $checkpoint")
  done
done

if [ ${#jobs[@]} -eq 0 ]; then
  echo "No sweep results found to evaluate."
  exit 1
fi

echo "Found ${#jobs[@]} configs to evaluate"
echo ""

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break

    read -r model_short task_idx suite env_task_id checkpoint <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}
    eval_dir="outputs/eval/act_${model_short}_libero_task${task_idx}"

    echo "[GPU $gpu] model=$model_short task=$task_idx suite=$suite env_task_id=$env_task_id -> $eval_dir"

    CUDA_VISIBLE_DEVICES=$gpu lerobot-eval \
      --policy.path="$checkpoint" \
      --env.type=libero \
      --env.task=$suite \
      --env.task_ids="[$env_task_id]" \
      --eval.batch_size=8 \
      --eval.n_episodes=100 \
      --output_dir="$eval_dir" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  i=$((i + NUM_GPUS))
done

echo ""
echo "All sweep evals complete."
