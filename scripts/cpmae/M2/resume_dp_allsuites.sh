#!/usr/bin/env bash
# Resume failed/incomplete DP runs from the multi-task all-suites sweep
#
# Only runs DP jobs that haven't reached 100k steps.
# Completed jobs (step >= 100k) are automatically skipped.
#
# 17 DP jobs to resume:
#   libero_10:     bs64 x {5e-5, 1e-5}            (2, from step 75k)
#   libero_spatial: bs64 x {1e-4, 5e-5, 1e-5}      (3, from step 25k-50k)
#   libero_object:  bs32 x 3 + bs64 x 3             (6, from step 0-25k)
#   libero_goal:    bs32 x 3 + bs64 x 3             (6, from step 0)
#
# Usage:
#   bash scripts/cpmae/resume_dp_allsuites.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/resume_dp_allsuites.sh 0 1 2 3            # 4 GPUs
#   DRY_RUN=1 bash scripts/cpmae/resume_dp_allsuites.sh 0 1      # print commands

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (must match original sweep) ───────────────────
STEPS=100000
EVAL_FREQ=0
SAVE_FREQ=25000
N_EVAL_EPISODES=20
EVAL_BATCH=10
BATCH_SIZES=(32 64)
LRS=(1e-4 5e-5 1e-5)
SEED=42
RESULTS_DIR="results/multitask_allsuites"
REPO_ID="HuggingFaceVLA/libero"

# ── Only DP, all 4 suites ─────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)

MAPPING_JSON="scripts/cpmae/task_mapping.json"

if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

# ── Pre-resolve per-suite metadata ────────────────────────────────
for suite in "${SUITES[@]}"; do
  num_tasks=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(m['suites']['$suite']['num_tasks'])")
  all_episodes=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$suite']['all_episodes']) + ']')")
  env_task_ids=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$suite']['env_task_ids']) + ']')")
  task_index_offset=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(min(m['suites']['$suite']['dataset_task_indices']))")

  export "SUITE_NUM_TASKS_${suite}=${num_tasks}"
  export "SUITE_EPISODES_${suite}=${all_episodes}"
  export "SUITE_ENV_IDS_${suite}=${env_task_ids}"
  export "SUITE_OFFSET_${suite}=${task_index_offset}"

  echo "  $suite: ${num_tasks} tasks, offset=${task_index_offset}"
done
echo ""

# ── run_task function ──────────────────────────────────────────────
run_task() {
    local suite="$1" bs="$2" lr="$3" job_seq="$4"

    # GPU assignment
    local num_gpus=${#GPUS[@]}
    local device_idx=$(( (job_seq - 1) % num_gpus ))
    local gpu=${GPUS[$device_idx]}
    export CUDA_VISIBLE_DEVICES="$gpu"

    # Per-suite metadata
    local num_tasks_var="SUITE_NUM_TASKS_${suite}"
    local episodes_var="SUITE_EPISODES_${suite}"
    local env_ids_var="SUITE_ENV_IDS_${suite}"
    local offset_var="SUITE_OFFSET_${suite}"
    local num_tasks="${!num_tasks_var}"
    local episodes="${!episodes_var}"
    local env_task_ids="${!env_ids_var}"
    local task_index_offset="${!offset_var}"

    local run_name="MT_dp_${suite}_bs${bs}_lr${lr}"
    local run_dir="${RESULTS_DIR}/${run_name}"

    # Skip if already completed
    local step_file="${run_dir}/checkpoints/last/training_state/training_step.json"
    if [[ -f "$step_file" ]]; then
        local saved_step
        saved_step=$(python3 -c "import json; print(json.load(open('$step_file'))['step'])")
        if [[ "$saved_step" -ge "$STEPS" ]]; then
            echo "[GPU ${gpu}] ${run_name} — completed (step ${saved_step}), skipping"
            return 0
        else
            echo "[GPU ${gpu}] ${run_name} — resuming from step ${saved_step}"
        fi
    else
        echo "[GPU ${gpu}] ${run_name} — starting fresh"
    fi

    local cmd=(
        lerobot-train
        --dataset.repo_id="$REPO_ID"
        --dataset.episodes="$episodes"
        --policy.type=diffusion
        --policy.vision_backbone=resnet18
        --policy.num_tasks="$num_tasks"
        --policy.task_embed_dim=64
        --policy.task_index_offset="$task_index_offset"
        --env.type=libero
        --env.task="$suite"
        --env.task_ids="$env_task_ids"
        --batch_size="$bs"
        --steps="$STEPS"
        --eval_freq="$EVAL_FREQ"
        --save_freq="$SAVE_FREQ"
        --eval.n_episodes="$N_EVAL_EPISODES"
        --eval.batch_size="$EVAL_BATCH"
        --seed="$SEED"
        --policy.optimizer_lr="$lr"
        --output_dir="$run_dir"
        --job_name="$run_name"
        --wandb.enable=true
        --wandb.project=cpmae_multitask
        --policy.push_to_hub=false
    )

    if [[ -n "${DRY_RUN:-}" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}
export -f run_task
export GPUS REPO_ID RESULTS_DIR STEPS EVAL_FREQ SAVE_FREQ
export N_EVAL_EPISODES EVAL_BATCH SEED

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#SUITES[@]} * ${#BATCH_SIZES[@]} * ${#LRS[@]} ))
echo "=== Resume DP sweep: ${#SUITES[@]} suites × ${#BATCH_SIZES[@]} bs × ${#LRS[@]} lr = ${TOTAL_JOBS} jobs (completed will be skipped) ==="
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {3} {#} \
        ::: "${SUITES[@]}" ::: "${BATCH_SIZES[@]}" ::: "${LRS[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs_resume" \
        -P "${PARALLEL}" \
        run_task {1} {2} {3} {#} \
        ::: "${SUITES[@]}" ::: "${BATCH_SIZES[@]}" ::: "${LRS[@]}"
fi

echo ""
echo "========================================="
echo "DP resume complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
