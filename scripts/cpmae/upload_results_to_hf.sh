#!/bin/bash
# Upload multitask_allsuites results from Vast.ai to Hugging Face Hub.
#
# Run ON the Vast instance:
#   bash /app/lerobot/scripts/cpmae/upload_results_to_hf.sh <HF_REPO_ID>
#
# Example:
#   bash /app/lerobot/scripts/cpmae/upload_results_to_hf.sh haitongma/cpmae-multitask-results
#
# What gets uploaded:
#   - checkpoints/{step}/pretrained_model/  (model weights)
#   - checkpoints/{step}/eval/              (eval results + videos)
#   - checkpoints/last                      (symlink)
#   - logs, resume.log
#
# What gets EXCLUDED:
#   - wandb/           (already synced to W&B)
#   - training_state/  (optimizer state, large, only for resuming)
#   - Incomplete runs  (no checkpoints dir or empty)

set -euo pipefail

REPO_ID="${1:?Usage: $0 <HF_REPO_ID> [--include-training-state]}"
INCLUDE_TRAINING_STATE=false
[[ "${2:-}" == "--include-training-state" ]] && INCLUDE_TRAINING_STATE=true

RESULTS_DIR="/app/results/multitask_allsuites"

# Check huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Run: pip install huggingface_hub"
    exit 1
fi

# Check login
if ! huggingface-cli whoami &>/dev/null; then
    echo "Not logged in to Hugging Face. Run:"
    echo "  huggingface-cli login --token <YOUR_TOKEN>"
    exit 1
fi

echo "=== Upload Plan ==="
echo "Source:  $RESULTS_DIR"
echo "HF repo: $REPO_ID (repo type: model)"
echo "Include training_state: $INCLUDE_TRAINING_STATE"
echo ""

# Skip incomplete runs (no checkpoints or empty checkpoints dir)
SKIP_RUNS=()
for run_dir in "$RESULTS_DIR"/MT_*; do
    run_name=$(basename "$run_dir")
    if [ ! -d "$run_dir/checkpoints" ] || [ -z "$(ls -A "$run_dir/checkpoints/" 2>/dev/null)" ]; then
        SKIP_RUNS+=("$run_name")
        echo "SKIP (incomplete): $run_name"
    fi
done
echo ""

# Build exclude patterns
EXCLUDE_ARGS="--exclude wandb/**"
if [ "$INCLUDE_TRAINING_STATE" = false ]; then
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude **/training_state/**"
fi
for skip in "${SKIP_RUNS[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude $skip/**"
done

# Estimate upload size
echo "Estimating upload size (excluding wandb + training_state + incomplete)..."
TOTAL_SIZE=0
for run_dir in "$RESULTS_DIR"/MT_*; do
    run_name=$(basename "$run_dir")
    # Skip incomplete
    skip=false
    for s in "${SKIP_RUNS[@]}"; do
        [[ "$run_name" == "$s" ]] && skip=true && break
    done
    $skip && continue

    if [ "$INCLUDE_TRAINING_STATE" = false ]; then
        size=$(du -sb "$run_dir" --exclude='wandb' --exclude='training_state' 2>/dev/null | awk '{print $1}')
    else
        size=$(du -sb "$run_dir" --exclude='wandb' 2>/dev/null | awk '{print $1}')
    fi
    TOTAL_SIZE=$((TOTAL_SIZE + size))
done
# Add logs
TOTAL_SIZE=$((TOTAL_SIZE + $(du -sb "$RESULTS_DIR/logs" "$RESULTS_DIR/logs_resume" "$RESULTS_DIR/resume.log" 2>/dev/null | awk '{s+=$1}END{print s}')))
echo "Estimated upload size: $((TOTAL_SIZE / 1024 / 1024 / 1024))G"
echo ""

read -rp "Proceed with upload? [y/N] " confirm
[[ "$confirm" != [yY] ]] && echo "Aborted." && exit 0

echo ""
echo "=== Uploading to $REPO_ID ==="
# shellcheck disable=SC2086
huggingface-cli upload "$REPO_ID" "$RESULTS_DIR" multitask_allsuites \
    --repo-type model \
    $EXCLUDE_ARGS

echo ""
echo "=== Done ==="
echo "View at: https://huggingface.co/$REPO_ID"
