#!/usr/bin/env bash
# Usage: ./cleanup_wandb_only.sh <results_dir>
# Finds subdirectories that contain ONLY a 'wandb' folder, shows them, and
# optionally deletes them after user confirmation.

set -euo pipefail

RESULTS_DIR="${1:-}"

if [[ -z "$RESULTS_DIR" ]]; then
    echo "Usage: $0 <results_dir>"
    echo "Example: $0 results/vit_sweep_allsuites"
    exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: directory '$RESULTS_DIR' not found."
    exit 1
fi

# Collect folders that contain exactly one entry and that entry is 'wandb'
wandb_only_dirs=()

while IFS= read -r -d '' subdir; do
    entries=("$subdir"/)
    # List all entries (files + dirs) in subdir
    mapfile -t entries < <(ls -1 "$subdir")
    if [[ ${#entries[@]} -eq 1 && "${entries[0]}" == "wandb" ]]; then
        wandb_only_dirs+=("$subdir")
    fi
done < <(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

if [[ ${#wandb_only_dirs[@]} -eq 0 ]]; then
    echo "No folders found that contain only a 'wandb' directory."
    exit 0
fi

echo "The following folders contain ONLY a 'wandb' directory:"
echo ""
for d in "${wandb_only_dirs[@]}"; do
    echo "  $d"
done
echo ""
echo "Total: ${#wandb_only_dirs[@]} folder(s)"
echo ""

read -rp "Delete all of the above folders? [y/N] " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    for d in "${wandb_only_dirs[@]}"; do
        rm -rf "$d"
        echo "Deleted: $d"
    done
    echo ""
    echo "Done. ${#wandb_only_dirs[@]} folder(s) deleted."
else
    echo "Aborted. Nothing was deleted."
fi
