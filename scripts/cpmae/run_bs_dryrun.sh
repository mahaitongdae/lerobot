#!/usr/bin/env bash
# Batch-size dry run: ACT + DP × bs={64,128,256} on one suite/backbone
# Runs 2k steps each, no eval, no save, no wandb. Reports OOM and timing.
#
# Usage:
#   bash scripts/cpmae/run_bs_dryrun.sh 0        # single GPU
#   bash scripts/cpmae/run_bs_dryrun.sh 0 1       # 2 GPUs (parallel)

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}

STEPS=2000
SUITE=libero_10
BACKBONE=imagenet
LR=5e-5
SEED=42
REPO_ID="HuggingFaceVLA/libero"
RESULTS_DIR="results/bs_dryrun"
MAPPING_JSON="scripts/cpmae/task_mapping.json"
BATCH_SIZES=(64 128 256)

if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

# Resolve suite metadata
NUM_TASKS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(m['suites']['$SUITE']['num_tasks'])")
EPISODES=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['all_episodes']) + ']')")
OFFSET=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(min(m['suites']['$SUITE']['dataset_task_indices']))")

mkdir -p "$RESULTS_DIR"

echo "=== Batch-size dry run ==="
echo "Suite: $SUITE ($NUM_TASKS tasks), backbone: $BACKBONE"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Steps: $STEPS (no eval, no save, no wandb)"
echo "GPUs: ${GPUS[*]}"
echo ""

# Pre-download dataset so timing isn't skewed
echo "Warming up dataset cache..."
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('$REPO_ID', episodes=$EPISODES)
print(f'  {len(ds.hf_dataset)} frames cached')
"
echo ""

JOB_IDX=0
PIDS=()
LOGS=()

for POLICY in act dp; do
  for BS in "${BATCH_SIZES[@]}"; do
    GPU_ID=${GPUS[$((JOB_IDX % NUM_GPUS))]}
    RUN_NAME="dryrun_${POLICY}_bs${BS}"
    RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
    LOG_FILE="${RESULTS_DIR}/${RUN_NAME}.log"

    # Build command
    CMD=(
      lerobot-train
      --dataset.repo_id="$REPO_ID"
      --dataset.episodes="$EPISODES"
      --policy.vision_backbone=resnet50
      --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1
      --policy.freeze_backbone=true
      --policy.num_tasks="$NUM_TASKS"
      --policy.task_embed_dim=64
      --policy.task_index_offset="$OFFSET"
      --batch_size="$BS"
      --steps="$STEPS"
      --eval_freq=0
      --save_freq=999999
      --seed="$SEED"
      --policy.optimizer_lr="$LR"
      --output_dir="$RUN_DIR"
      --job_name="$RUN_NAME"
      --wandb.enable=false
      --policy.push_to_hub=false
    )

    if [[ "$POLICY" == "act" ]]; then
      CMD+=(--policy.type=act --policy.optimizer_lr_backbone="$LR")
    else
      CMD+=(--policy.type=diffusion --policy.use_group_norm=false)
    fi

    echo "[GPU $GPU_ID] $RUN_NAME — launching..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    LOGS+=("$LOG_FILE")

    JOB_IDX=$((JOB_IDX + 1))

    # If all GPUs busy, wait for the batch to finish before launching more
    if (( JOB_IDX % NUM_GPUS == 0 )); then
      for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
      PIDS=()
    fi
  done
done

# Wait for any remaining jobs
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

echo ""
echo "========================================="
echo "Results"
echo "========================================="
printf "%-25s %6s %10s %10s %s\n" "Run" "BS" "Time(s)" "Est 100k" "Status"
echo "---------------------------------------------------------------------------"

for POLICY in act dp; do
  for BS in "${BATCH_SIZES[@]}"; do
    RUN_NAME="dryrun_${POLICY}_bs${BS}"
    LOG_FILE="${RESULTS_DIR}/${RUN_NAME}.log"

    # Check for OOM
    if grep -qi "out of memory\|CUDA error\|OOM" "$LOG_FILE" 2>/dev/null; then
      STATUS="OOM"
      ELAPSED="-"
      EST100K="-"
    elif grep -q "step.*$STEPS\b" "$LOG_FILE" 2>/dev/null || grep -qi "Training complete\|done" "$LOG_FILE" 2>/dev/null; then
      # Extract wall-clock time from log (look for step timing)
      # lerobot logs: "step:2000 smpl:... ep:... epch:... loss:... grads:... lr:... ... dt:0.123s wall:1234.5s"
      WALL=$(grep -oP 'wall:\K[0-9.]+' "$LOG_FILE" | tail -1)
      if [[ -n "$WALL" ]]; then
        ELAPSED="$WALL"
        EST100K=$(python3 -c "t=$WALL; print(f'{t/$STEPS*100000/3600:.1f}h')")
      else
        # Fallback: extract timestamps from first and last step lines
        ELAPSED=$(python3 -c "
import re, sys
lines = [l for l in open('$LOG_FILE') if 'step' in l.lower()]
if not lines:
    print('-'); sys.exit()
# Try to find elapsed time pattern
for l in reversed(lines):
    m = re.search(r'(\d+\.\d+)\s*s(?:ec)?', l)
    if m:
        print(m.group(1)); sys.exit()
print('-')
")
        EST100K="-"
      fi
      STATUS="OK"
    else
      STATUS="FAIL (check log)"
      ELAPSED="-"
      EST100K="-"
    fi

    printf "%-25s %6s %10s %10s %s\n" "$RUN_NAME" "$BS" "${ELAPSED}s" "$EST100K" "$STATUS"
  done
done

echo ""
echo "Logs: $RESULTS_DIR/*.log"
echo "To inspect a specific run:  tail -30 $RESULTS_DIR/dryrun_<policy>_bs<N>.log"

# GPU memory summary from logs
echo ""
echo "Peak GPU memory (from logs, if available):"
for POLICY in act dp; do
  for BS in "${BATCH_SIZES[@]}"; do
    LOG_FILE="${RESULTS_DIR}/dryrun_${POLICY}_bs${BS}.log"
    MEM=$(grep -oiP '(peak|max|allocated).*?(\d+\.?\d*)\s*(MiB|GiB|MB|GB)' "$LOG_FILE" 2>/dev/null | tail -1)
    if [[ -z "$MEM" ]]; then
      MEM=$(grep -oiP '\d+\.?\d*\s*(MiB|GiB)' "$LOG_FILE" 2>/dev/null | tail -1)
    fi
    printf "  %-25s %s\n" "dryrun_${POLICY}_bs${BS}" "${MEM:-not reported}"
  done
done
