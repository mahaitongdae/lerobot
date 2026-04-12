#!/bin/bash
TASK_INDEX=${1:?Usage: $0 <task_index>}

BATCH_SIZES=(16 32 64 128)
LRS=(1e-5 5e-5 1e-4 5e-4)

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

jobs=()
for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LRS[@]}"; do
    jobs+=("$bs $lr")
  done
done

echo "Sweeping ${#jobs[@]} configs across $NUM_GPUS GPUs for task $TASK_INDEX"

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge ${#jobs[@]} ] && break
    read -r bs lr <<< "${jobs[$idx]}"
    gpu=${GPUS[$g]}
    echo "Launching bs=$bs lr=$lr on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/train_act_libero.sh "$TASK_INDEX" "$bs" "$lr" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  i=$((i + NUM_GPUS))
done

echo "All sweep jobs complete."
