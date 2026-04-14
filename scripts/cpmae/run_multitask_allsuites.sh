#!/usr/bin/env bash
# Multi-task HP sweep: ACT and DP on all 4 LIBERO suites (10 tasks each)
#
# Grid: 2 policies × 4 suites × 3 batch_sizes × 3 lrs = 72 jobs
# Suites: libero_10, libero_spatial, libero_object, libero_goal
# Total: 72 runs
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/run_multitask_allsuites.sh 0 1 2 3 4 5 6 7   # 8 GPUs (all parallel)
#   bash scripts/cpmae/run_multitask_allsuites.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/run_multitask_allsuites.sh 0                  # single GPU (sequential)
#   DRY_RUN=1 bash scripts/cpmae/run_multitask_allsuites.sh 0 1      # print commands only
#
# Estimated: ~100 GPU-hours total (72 × ~1.4h each at 100k steps)

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (fixed) ────────────────────────────────────────
STEPS=100000
EVAL_FREQ=0
SAVE_FREQ=25000
N_EVAL_EPISODES=20
EVAL_BATCH=10
BATCH_SIZES=(32 64 128)
LRS=(1e-4 5e-5 1e-5)
SEED=42
RESULTS_DIR="results/multitask_allsuites"
REPO_ID="HuggingFaceVLA/libero"

# ── Suite list ─────────────────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
POLICIES=(act dp)

MAPPING_JSON="scripts/cpmae/task_mapping.json"

if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

# ── Pre-resolve per-suite metadata from task_mapping.json ──────────
# Flatten into individual exported vars (associative arrays can't be exported).
for suite in "${SUITES[@]}"; do
  num_tasks=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(m['suites']['$suite']['num_tasks'])")
  all_episodes=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$suite']['all_episodes']) + ']')")
  env_task_ids=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$suite']['env_task_ids']) + ']')")
  task_index_offset=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(min(m['suites']['$suite']['dataset_task_indices']))")

  export "SUITE_NUM_TASKS_${suite}=${num_tasks}"
  export "SUITE_EPISODES_${suite}=${all_episodes}"
  export "SUITE_ENV_IDS_${suite}=${env_task_ids}"
  export "SUITE_OFFSET_${suite}=${task_index_offset}"

  echo "  $suite: ${num_tasks} tasks, offset=${task_index_offset}, episodes=$(echo "$all_episodes" | tr -cd ',' | wc -c | xargs)+1"
done
echo ""

# ── run_task function ──────────────────────────────────────────────
run_task() {
    local policy="$1" suite="$2" bs="$3" lr="$4" job_seq="$5"

    # GPU assignment via job sequence number (unique, never recycled)
    local num_gpus=${#GPUS[@]}
    local device_idx=$(( (job_seq - 1) % num_gpus ))
    local gpu=${GPUS[$device_idx]}
    export CUDA_VISIBLE_DEVICES="$gpu"

    # Reconstruct per-suite metadata from flattened exports
    local num_tasks_var="SUITE_NUM_TASKS_${suite}"
    local episodes_var="SUITE_EPISODES_${suite}"
    local env_ids_var="SUITE_ENV_IDS_${suite}"
    local offset_var="SUITE_OFFSET_${suite}"
    local num_tasks="${!num_tasks_var}"
    local episodes="${!episodes_var}"
    local env_task_ids="${!env_ids_var}"
    local task_index_offset="${!offset_var}"

    # Build run name
    local run_name="MT_${policy}_${suite}_bs${bs}_lr${lr}"
    local run_dir="${RESULTS_DIR}/${run_name}"

    # Checkpoint skip
    if [[ -d "${run_dir}/checkpoints/last/pretrained_model" ]]; then
        echo "[GPU ${gpu}] ${run_name} — completed, skipping"
        return 0
    fi

    # Build command
    local cmd=()
    if [[ "$policy" == "act" ]]; then
        cmd=(
            lerobot-train
            --dataset.repo_id="$REPO_ID"
            --dataset.episodes="$episodes"
            --policy.type=act
            --policy.vision_backbone=resnet18
            --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
            --policy.freeze_backbone=true
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
            --policy.optimizer_lr_backbone="$lr"
            --output_dir="$run_dir"
            --job_name="$run_name"
            --wandb.enable=true
            --wandb.project=cpmae_multitask
            --policy.push_to_hub=false
        )
    elif [[ "$policy" == "dp" ]]; then
        cmd=(
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
    fi

    echo "[GPU ${gpu}] ${run_name} (${policy}, ${suite}, ${num_tasks} tasks, offset=${task_index_offset})"

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
TOTAL_JOBS=$(( ${#POLICIES[@]} * ${#SUITES[@]} * ${#BATCH_SIZES[@]} * ${#LRS[@]} ))
echo "=== Multi-task HP sweep: ${#POLICIES[@]} policies × ${#SUITES[@]} suites × ${#BATCH_SIZES[@]} bs × ${#LRS[@]} lr = ${TOTAL_JOBS} jobs ==="
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {3} {4} {#} \
        ::: "${POLICIES[@]}" ::: "${SUITES[@]}" \
        ::: "${BATCH_SIZES[@]}" ::: "${LRS[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {3} {4} {#} \
        ::: "${POLICIES[@]}" ::: "${SUITES[@]}" \
        ::: "${BATCH_SIZES[@]}" ::: "${LRS[@]}"
fi

echo ""
echo "========================================="
echo "Multi-task all-suites complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
