#!/usr/bin/env bash
# Add a `last` symlink under each run's checkpoints folder,
# pointing to the highest-numbered checkpoint.
#
# Usage:
#   ./scripts/add_last_symlink.sh <results_subfolder> [run1 run2 ...]
#
# Examples:
#   ./scripts/add_last_symlink.sh results/ssl_sweep_allsuites
#   ./scripts/add_last_symlink.sh results/ssl_sweep_allsuites SSL_dp_libero_spatial_byol SSL_dp_libero_spatial_simclr

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <results_subfolder> [run1 run2 ...]" >&2
  exit 1
fi

base_dir="$1"
shift

if [[ ! -d "$base_dir" ]]; then
  echo "Error: directory not found: $base_dir" >&2
  exit 1
fi

cd "$base_dir"

if [[ $# -gt 0 ]]; then
  runs=("$@")
else
  runs=()
  for d in */; do
    run="${d%/}"
    [[ -d "$run/checkpoints" ]] && runs+=("$run")
  done
fi

if [[ ${#runs[@]} -eq 0 ]]; then
  echo "No runs with a checkpoints/ folder found in $base_dir" >&2
  exit 1
fi

for run in "${runs[@]}"; do
  ckpt_dir="$run/checkpoints"
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "Skipping $run (no $ckpt_dir)" >&2
    continue
  fi

  latest=$(ls "$ckpt_dir" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
  if [[ -z "$latest" ]]; then
    echo "Skipping $run (no numeric checkpoints in $ckpt_dir)" >&2
    continue
  fi

  ln -sfn "$latest" "$ckpt_dir/last"
  echo "$run: last -> $latest"
done
