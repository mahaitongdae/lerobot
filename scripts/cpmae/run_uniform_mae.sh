#!/bin/bash
# Pretrain Uniform MAE encoder (standalone, matches run_cpmae.sh R201 setup)
#
# Usage:
#   bash scripts/cpmae/run_uniform_mae.sh 0   # GPU index

set -euo pipefail

GPU=${1:-0}

RESULTS_DIR="results/M3_cpmae"
UMAE_DIR="$RESULTS_DIR/R201_uniform_mae"

mkdir -p "$UMAE_DIR"

if [ -f "$UMAE_DIR/encoder_final.pt" ]; then
  echo "R201 Uniform MAE already exists at $UMAE_DIR/encoder_final.pt"
  exit 0
fi

echo "Training Uniform MAE encoder on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/cpmae/pretrain_mae.py \
  --mode=uniform \
  --output_dir="$UMAE_DIR" \
  --epochs=400 \
  --batch_size=256 \
  --lr=1.5e-4 \
  --uniform_mask_ratio=0.75 \
  --gpu=0

echo "R201 Uniform MAE DONE"
