#!/bin/bash
# M3b-improve: CP-MAE improvement sweep (4 variants, 4 GPUs)
#
# Variants:
#   R300: CP-MAE + per-patch norm                        (improvement #1)
#   R301: CP-MAE + per-patch norm + loss_weight=1.0      (improvement #3)
#   R302: CP-MAE + per-patch norm + ViT-B/16             (improvement #4)
#   R303: Hybrid masking + per-patch norm                 (improvement #6)
#
# All use same base HP as M3a (400 epochs, bs=256, lr=1.5e-4).
# Per-patch normalization (#1) is applied to all variants as a baseline improvement.
#
# Prerequisites:
#   - Contact labels: results/contact_labels/ (from contact_detector.py)
#
# Usage:
#   bash scripts/cpmae/M3/run_cpmae_improved.sh 0 1 2 3   # 4 GPUs (parallel)
#   bash scripts/cpmae/M3/run_cpmae_improved.sh 0          # single GPU (sequential)
#   DRY_RUN=1 bash scripts/cpmae/M3/run_cpmae_improved.sh 0 1 2 3  # print only

set -euo pipefail

GPUS=("${@:-0}")
NUM_GPUS=${#GPUS[@]}

RESULTS_DIR="results/M3_cpmae_improved"
CONTACT_LABELS_DIR="results/contact_labels"

EPOCHS=400
BATCH_SIZE=256
LR=1.5e-4
CONTACT_MASK_RATIO=0.90
TRANSIT_MASK_RATIO=0.75

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

cleanup() {
  echo ""; echo "Caught interrupt — killing background jobs..."
  kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; exit 1
}
trap cleanup SIGINT SIGTERM

# Ensure contact labels exist
if [ ! -d "$CONTACT_LABELS_DIR" ] || [ -z "$(ls -A $CONTACT_LABELS_DIR 2>/dev/null)" ]; then
  echo "Generating contact labels..."
  python3 scripts/cpmae/contact_detector.py --all_tasks --output_dir="$CONTACT_LABELS_DIR"
fi

# ── Experiment definitions ──────────────────────────────────────────
# Format: RUN_ID  MODE  EXTRA_ARGS...
declare -a EXPERIMENTS
EXPERIMENTS=(
  "R300 cpmae --normalize_target --contact_loss_weight=2.0"
  "R301 cpmae --normalize_target --contact_loss_weight=1.0"
  "R302 cpmae --normalize_target --contact_loss_weight=2.0 --encoder_dim=768 --encoder_heads=12 --decoder_dim=384 --decoder_heads=6"
  "R303 hybrid --normalize_target --contact_loss_weight=2.0 --hybrid_gripper_rows=4"
)

echo "=== M3b-improve: ${#EXPERIMENTS[@]} variants ==="
echo "GPUs: ${GPUS[*]}"
echo ""

run_experiment() {
  local idx="$1"
  local line="${EXPERIMENTS[$idx]}"

  # Parse RUN_ID and MODE (first two tokens), rest are extra args
  local run_id mode extra
  run_id=$(echo "$line" | awk '{print $1}')
  mode=$(echo "$line" | awk '{print $2}')
  extra=$(echo "$line" | awk '{$1=""; $2=""; print $0}' | sed 's/^ *//')

  local gpu_idx=$(( idx % NUM_GPUS ))
  local gpu=${GPUS[$gpu_idx]}

  local run_name="${run_id}_${mode}"
  local run_dir="${RESULTS_DIR}/${run_name}"

  # Skip if already done
  if [[ -f "${run_dir}/encoder_final.pt" ]]; then
    echo "[GPU ${gpu}] ${run_name} — already complete, skipping"
    return 0
  fi

  echo "[GPU ${gpu}] ${run_name} (mode=${mode}, ${extra})"

  local cmd=(
    python3 scripts/cpmae/pretrain_mae.py
    --mode="$mode"
    --output_dir="$run_dir"
    --epochs="$EPOCHS"
    --batch_size="$BATCH_SIZE"
    --lr="$LR"
    --contact_mask_ratio="$CONTACT_MASK_RATIO"
    --transit_mask_ratio="$TRANSIT_MASK_RATIO"
    --contact_labels_dir="$CONTACT_LABELS_DIR"
    --gpu=0
  )

  # Append extra args
  # shellcheck disable=SC2206
  cmd+=($extra)

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "${RESULTS_DIR}/logs/${run_name}.log"
}

# ── Dispatch: launch up to NUM_GPUS experiments in parallel ─────────
TOTAL=${#EXPERIMENTS[@]}
FAILED_JOBS=0

i=0
while [ $i -lt $TOTAL ]; do
  pids=()
  job_names=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    idx=$((i + g))
    [ $idx -ge $TOTAL ] && break
    run_experiment "$idx" &
    pids+=($!)
    local_name=$(echo "${EXPERIMENTS[$idx]}" | awk '{print $1"_"$2}')
    job_names+=("$local_name")
  done
  for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then
      echo "FAILED: ${job_names[$j]}"
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
  done
  i=$((i + NUM_GPUS))
done

echo ""
echo "========================================="
echo "M3b-improve: CP-MAE improvement sweep complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Variants:"
echo "  R300: CP-MAE + per-patch norm (baseline improvement)"
echo "  R301: CP-MAE + per-patch norm + loss_weight=1.0"
echo "  R302: CP-MAE + per-patch norm + ViT-B/16 (768d, 12 heads)"
echo "  R303: Hybrid masking + per-patch norm"
echo ""
if [ $FAILED_JOBS -gt 0 ]; then
  echo "WARNING: $FAILED_JOBS job(s) failed — check logs"
fi
echo "Next: run downstream ACT training with these encoders."

exit $( [ $FAILED_JOBS -gt 0 ] && echo 1 || echo 0 )
