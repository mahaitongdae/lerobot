#!/usr/bin/env bash
# Remove intermediate checkpoints, keeping only "last" (the symlink target) and "best".
#
# Usage:
#   bash scripts/cpmae/cleanup_checkpoints.sh results/M5_multi_encoder
#   bash scripts/cpmae/cleanup_checkpoints.sh results/M5_multi_encoder --dry-run
#   bash scripts/cpmae/cleanup_checkpoints.sh results/M5_multi_encoder/M5_dp_libero_goal_dino3p_siglip_vitb_wrist

set -euo pipefail

DRY_RUN=false
TARGETS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) TARGETS+=("$arg") ;;
    esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Usage: $0 <results_dir> [--dry-run]"
    echo "  results_dir can be a single run dir or a parent containing multiple runs."
    exit 1
fi

cleanup_run() {
    local ckpt_dir="$1/checkpoints"

    if [[ ! -d "$ckpt_dir" ]]; then
        return 0
    fi

    local run_name
    run_name=$(basename "$1")

    local has_last=false
    local has_best=false
    local last_target=""

    if [[ -L "$ckpt_dir/last" ]]; then
        has_last=true
        last_target=$(readlink -f "$ckpt_dir/last")
    fi
    [[ -d "$ckpt_dir/best" ]] && has_best=true

    if ! $has_last && ! $has_best; then
        echo "WARNING: $run_name — no 'last' symlink or 'best' directory found, skipping cleanup"
        return 0
    fi
    if ! $has_last; then
        echo "WARNING: $run_name — no 'last' symlink found (only 'best' will be kept)"
    fi
    if ! $has_best; then
        echo "WARNING: $run_name — no 'best' directory found (only 'last' will be kept)"
    fi

    local removed=0
    local kept=0

    for entry in "$ckpt_dir"/*/; do
        [[ -d "$entry" ]] || continue
        local name
        name=$(basename "$entry")

        # Keep "best" directory
        if [[ "$name" == "best" ]]; then
            kept=$((kept + 1))
            continue
        fi

        # Keep "last" symlink itself (not a real dir, but just in case)
        if [[ "$name" == "last" ]]; then
            continue
        fi

        # Keep the directory that "last" points to
        local real_entry
        real_entry=$(readlink -f "$entry")
        if [[ -n "$last_target" && "$real_entry" == "$last_target" ]]; then
            kept=$((kept + 1))
            continue
        fi

        # Remove this intermediate checkpoint
        if $DRY_RUN; then
            echo "  [dry-run] would remove: $entry"
        else
            rm -rf "$entry"
        fi
        removed=$((removed + 1))
    done

    if [[ $removed -gt 0 || $kept -gt 0 ]]; then
        local verb="removed"
        $DRY_RUN && verb="would remove"
        echo "$run_name: $verb $removed checkpoint(s), kept $kept (last + best)"
    fi
}

for target in "${TARGETS[@]}"; do
    # If target has a checkpoints/ dir directly, it's a single run
    if [[ -d "$target/checkpoints" ]]; then
        cleanup_run "$target"
    else
        # Parent dir containing multiple runs
        for run_dir in "$target"/*/; do
            [[ -d "$run_dir/checkpoints" ]] && cleanup_run "$run_dir"
        done
    fi
done
