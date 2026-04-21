#!/bin/bash
# Upload results from Vast.ai to Hugging Face Hub.
#
# Run ON the Vast instance:
#   bash /app/lerobot/scripts/cpmae/upload_results_to_hf.sh <RESULTS_DIR> <HF_REPO_ID> [--include-training-state] [--dry-run]
#
# Example:
#   bash ./scripts/cpmae/upload_results_to_hf.sh \
#       ./results/multitask_allsuites haitong-ma/cpmae-multitask-results
#
# Flags:
#   --include-training-state  Also upload optimizer state dirs.
#   --dry-run                 Query the HF repo and print only the delta
#                             (files missing or size-mismatched). Does not upload.
#
# What gets uploaded:
#   - checkpoints/{step}/pretrained_model/  (model weights)
#   - checkpoints/{step}/eval/              (eval results + videos)
#   - checkpoints/last                      (symlink)
#   - logs, resume.log
#
# What gets EXCLUDED:
#   - wandb/           (already synced to W&B)
#   - .cache/          (local HF upload cache, not needed on hub)
#   - training_state/  (optimizer state, large, only for resuming)
#   - Incomplete runs  (no checkpoints dir or empty)

set -euo pipefail

USAGE="Usage: $0 <RESULTS_DIR> <HF_REPO_ID> [--include-training-state] [--dry-run]"
RESULTS_DIR="${1:?$USAGE}"
REPO_ID="${2:?$USAGE}"
INCLUDE_TRAINING_STATE=false
DRY_RUN=false
for arg in "${@:3}"; do
    case "$arg" in
        --include-training-state) INCLUDE_TRAINING_STATE=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown flag: $arg"; echo "$USAGE"; exit 1 ;;
    esac
done

RESULTS_DIR="${RESULTS_DIR%/}"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: results dir not found: $RESULTS_DIR"
    exit 1
fi

# Check hf CLI is available
if ! command -v hf &>/dev/null; then
    echo "ERROR: hf not found. Run: pip install -U huggingface_hub"
    exit 1
fi

# Check login
if ! hf auth whoami &>/dev/null; then
    echo "Not logged in to Hugging Face. Run:"
    echo "  hf auth login --token <YOUR_TOKEN>"
    exit 1
fi

echo "=== Upload Plan ==="
echo "Source:  $RESULTS_DIR"
echo "HF repo: $REPO_ID (repo type: model)"
echo "Include training_state: $INCLUDE_TRAINING_STATE"
echo "Dry run: $DRY_RUN"
echo ""

# Skip incomplete runs (no checkpoints or empty checkpoints dir)
SKIP_RUNS=()
for run_dir in "$RESULTS_DIR"/*; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    if [ ! -d "$run_dir/checkpoints" ] || [ -z "$(ls -A "$run_dir/checkpoints/" 2>/dev/null)" ]; then
        SKIP_RUNS+=("$run_name")
        echo "SKIP (incomplete): $run_name"
    fi
done
echo ""

# Build exclude patterns
EXCLUDE_ARGS="--exclude wandb/** --exclude **/.cache/**"
if [ "$INCLUDE_TRAINING_STATE" = false ]; then
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude **/training_state/**"
fi
for skip in "${SKIP_RUNS[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude $skip/**"
done

# Estimate upload size
echo "Estimating upload size (excluding wandb + .cache + training_state + incomplete)..."
TOTAL_SIZE=0
for run_dir in "$RESULTS_DIR"/*; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    # Skip incomplete
    skip=false
    if [ ${#SKIP_RUNS[@]} -gt 0 ]; then
        for s in "${SKIP_RUNS[@]}"; do
            [[ "$run_name" == "$s" ]] && skip=true && break
        done
    fi
    $skip && continue

    if [ "$INCLUDE_TRAINING_STATE" = false ]; then
        size=$(du -sb "$run_dir" --exclude='wandb' --exclude='.cache' --exclude='training_state' 2>/dev/null | awk '{print $1}' || true)
    else
        size=$(du -sb "$run_dir" --exclude='wandb' --exclude='.cache' 2>/dev/null | awk '{print $1}' || true)
    fi
    TOTAL_SIZE=$((TOTAL_SIZE + ${size:-0}))
done
# Add top-level logs/resume files if present
EXTRA_PATHS=()
for p in "$RESULTS_DIR/logs" "$RESULTS_DIR/logs_resume" "$RESULTS_DIR/resume.log"; do
    [ -e "$p" ] && EXTRA_PATHS+=("$p")
done
if [ ${#EXTRA_PATHS[@]} -gt 0 ]; then
    extra=$(du -sb "${EXTRA_PATHS[@]}" 2>/dev/null | awk '{s+=$1}END{print s+0}' || true)
    TOTAL_SIZE=$((TOTAL_SIZE + ${extra:-0}))
fi
echo "Estimated upload size: $((TOTAL_SIZE / 1024 / 1024 / 1024))G (local total; HF will dedup already-uploaded blobs)"
echo ""

# Dry run: compute delta against existing HF repo contents
if [ "$DRY_RUN" = true ]; then
    echo "=== Dry run: computing delta vs $REPO_ID ==="
    SKIP_RUNS_CSV=$(IFS=,; echo "${SKIP_RUNS[*]:-}")
    RESULTS_DIR="$RESULTS_DIR" \
    REPO_ID="$REPO_ID" \
    INCLUDE_TRAINING_STATE="$INCLUDE_TRAINING_STATE" \
    SKIP_RUNS_CSV="$SKIP_RUNS_CSV" \
    python - <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

results_dir = Path(os.environ["RESULTS_DIR"]).resolve()
repo_id = os.environ["REPO_ID"]
include_ts = os.environ["INCLUDE_TRAINING_STATE"] == "true"
skip_runs = {s for s in os.environ.get("SKIP_RUNS_CSV", "").split(",") if s}

EXCLUDE_DIR_NAMES = {"wandb", ".cache"}
if not include_ts:
    EXCLUDE_DIR_NAMES.add("training_state")


def is_excluded(rel: Path) -> bool:
    parts = rel.parts
    if parts and parts[0] in skip_runs:
        return True
    return any(p in EXCLUDE_DIR_NAMES for p in parts)


print(f"Scanning local: {results_dir}")
local: dict[str, int] = {}
for root, dirs, files in os.walk(results_dir, followlinks=False):
    root_path = Path(root)
    rel_root = root_path.relative_to(results_dir)
    # Prune excluded directories early
    dirs[:] = [d for d in dirs if not is_excluded(rel_root / d)]
    if rel_root != Path(".") and is_excluded(rel_root):
        continue
    for f in files:
        rel = rel_root / f
        if is_excluded(rel):
            continue
        fp = root_path / f
        try:
            local[str(rel)] = fp.stat().st_size
        except OSError:
            pass

print(f"Local files considered: {len(local)}  ({sum(local.values()) / 1e9:.2f} GB)")

api = HfApi()
remote: dict[str, int] = {}
try:
    for info in api.list_repo_tree(repo_id=repo_id, recursive=True, repo_type="model"):
        # Only regular/LFS files have a size; folders have type == 'directory'
        if getattr(info, "type", None) == "file":
            remote[info.path] = int(getattr(info, "size", 0) or 0)
except RepositoryNotFoundError:
    print(f"Repo {repo_id} does not exist yet; entire local selection will be uploaded.")

print(f"Remote files: {len(remote)}  ({sum(remote.values()) / 1e9:.2f} GB)")

missing: list[tuple[str, int]] = []
mismatched: list[tuple[str, int, int]] = []
for path, lsize in local.items():
    if path not in remote:
        missing.append((path, lsize))
    elif remote[path] != lsize:
        mismatched.append((path, lsize, remote[path]))

delta = sum(s for _, s in missing) + sum(ls for _, ls, _ in mismatched)
print()
print(f"Missing on remote: {len(missing)} files  ({sum(s for _, s in missing) / 1e9:.2f} GB)")
print(f"Size-mismatched:   {len(mismatched)} files  ({sum(ls for _, ls, _ in mismatched) / 1e9:.2f} GB)")
print(f"Total delta:       {delta / 1e9:.2f} GB")

def show(title: str, items: list, fmt):
    if not items:
        return
    print(f"\n{title} (top 20 by size):")
    for row in sorted(items, key=lambda r: -(r[1] if len(r) == 2 else r[1]))[:20]:
        print("  " + fmt(row))

show("Missing", missing, lambda r: f"{r[1] / 1e6:>10.1f} MB  {r[0]}")
show(
    "Mismatched",
    mismatched,
    lambda r: f"local={r[1] / 1e6:>9.1f} MB  remote={r[2] / 1e6:>9.1f} MB  {r[0]}",
)
PY
    echo ""
    echo "(Dry run only — no upload performed.)"
    exit 0
fi

read -rp "Proceed with upload? [y/N] " confirm
[[ "$confirm" != [yY] ]] && echo "Aborted." && exit 0

echo ""
echo "=== Uploading to $REPO_ID ==="
# shellcheck disable=SC2086
hf upload-large-folder "$REPO_ID" "$RESULTS_DIR" \
    --repo-type model \
    $EXCLUDE_ARGS

echo ""
echo "=== Done ==="
echo "View at: https://huggingface.co/$REPO_ID"
