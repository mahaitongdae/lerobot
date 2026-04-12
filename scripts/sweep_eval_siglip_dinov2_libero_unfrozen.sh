#!/bin/bash
# Sweep evaluation: ACT + SigLIP-base (unfrozen) and ACT + DINOv2-base (unfrozen)
# across 4 tasks on GPUs. Evaluates all checkpoint iterations for each.

set -euo pipefail

TASK_INDICES=(0 1 2 3)
ITERS=(020000 040000 060000 080000 100000)
GPUS=(1 2)
NUM_GPUS=${#GPUS[@]}

BASE_TRAIN_DIR="/mnt/shared/haitongma/wdir/lerobot/outputs/train"
declare -A TRAIN_DIRS

MODELS=(
  "dinov2-base"
  # "siglip-base-patch16-224"
)

# Discover latest matching training output dir for each (model, task) combo
for model_short in "${MODELS[@]}"; do
  for task_idx in 0 1 2 3; do
    matches=( "$BASE_TRAIN_DIR"/act_${model_short}_libero_task${task_idx}_*_bblr1e-6_unfrozen_* )
    for m in "${matches[@]}"; do
      [ -d "$m" ] && TRAIN_DIRS["${model_short},${task_idx}"]="$m"
    done
  done
done

cleanup() {
  echo ""
  echo "Caught interrupt — killing all background jobs..."
  kill $(jobs -p) 2>/dev/null
  wait 2>/dev/null
  echo "All jobs stopped."
  exit 1
}
trap cleanup SIGINT SIGTERM

# Hardcoded task mapping: dataset task_index -> "suite env_task_id"
# Resolved from HuggingFaceVLA/libero metadata + LIBERO benchmark suites
declare -A TASK_MAP
TASK_MAP[0]="libero_10 4"   # put the white mug on the left plate and put the yellow and white mug on the right plate
TASK_MAP[1]="libero_10 6"   # put the white mug on the plate and put the chocolate pudding to the right of the plate
TASK_MAP[2]="libero_10 9"   # put the yellow and white mug in the microwave and close it
TASK_MAP[3]="libero_10 2"   # turn on the stove and put the moka pot on it

jobs=()
for model_short in "${MODELS[@]}"; do
  for task_idx in "${TASK_INDICES[@]}"; do
    key="${model_short},${task_idx}"
    dir="${TRAIN_DIRS[$key]:-}"
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
      echo "SKIP: no training output for model=$model_short task=$task_idx"
      continue
    fi

    read -r suite env_task_id <<< "${TASK_MAP[$task_idx]}"

    for iter in "${ITERS[@]}"; do
      checkpoint="$dir/checkpoints/${iter}/pretrained_model"
      if [ ! -d "$checkpoint" ]; then
        echo "SKIP: no checkpoint at $checkpoint"
        continue
      fi
      jobs+=("$model_short $task_idx $suite $env_task_id $iter $checkpoint")
    done
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

    read -r model_short task_idx suite env_task_id iter checkpoint <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}
    eval_dir="/mnt/shared/haitongma/wdir/lerobot/outputs/eval/act_${model_short}_libero_task${task_idx}_bblr1e-6_unfrozen/${iter}"

    echo "[GPU $gpu] model=$model_short task=$task_idx iter=$iter suite=$suite env_task_id=$env_task_id -> $eval_dir"

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
