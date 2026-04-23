#!/usr/bin/env bash
# M3b (aligned): CP-MAE / Uniform-MAE downstream ACT training on all 4 LIBERO
# suites, with hyperparameters matched to run_vit_sweep_allsuites.sh so results
# are directly comparable to the ViT backbone sweep.
#
# Variants (all ViT encoders, 224x224):
#   cpmae_frozen — CP-MAE encoder, frozen
#   cpmae_ft     — CP-MAE encoder, fine-tuned
#   umae_frozen  — Uniform-MAE encoder, frozen
#   umae_ft      — Uniform-MAE encoder, fine-tuned
#
# Grid: 1 policy (ACT) × 4 suites × 4 variants = 16 jobs
# Suites: libero_10, libero_spatial, libero_object, libero_goal
#
# Prerequisites:
#   1. Pretrained encoders from M3a (run_cpmae.sh phase 1):
#        results/M3_cpmae/R200_cpmae/encoder_final.pt
#        results/M3_cpmae/R201_uniform_mae/encoder_final.pt
#   2. python scripts/cpmae/build_task_mapping.py   # one-time
#
# Usage:
#   bash scripts/cpmae/run_cpmae_sweep_allsuites.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/run_cpmae_sweep_allsuites.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/run_cpmae_sweep_allsuites.sh 0                  # single GPU
#   DRY_RUN=1 bash scripts/cpmae/run_cpmae_sweep_allsuites.sh 0 1      # print only

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (aligned with run_vit_sweep_allsuites.sh) ─────
STEPS=100000
EVAL_FREQ=25000
SAVE_FREQ=25000
N_EVAL_EPISODES=20
EVAL_BATCH=20
BATCH_SIZE=64
LR=5e-5
SEED=42
RESULTS_DIR="results/cpmae_sweep_allsuites"
REPO_ID="HuggingFaceVLA/libero"

# ── Pretrained encoder checkpoints (from M3a / run_cpmae.sh phase 1) ─────
M3A_DIR="${M3A_DIR:-results/M3_cpmae}"
CPMAE_CKPT="${M3A_DIR}/R200_cpmae/encoder_final.pt"
UMAE_CKPT="${M3A_DIR}/R201_uniform_mae/encoder_final.pt"

for ckpt in "$CPMAE_CKPT" "$UMAE_CKPT"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: encoder checkpoint not found: $ckpt" >&2
    echo "Run M3a first: bash scripts/cpmae/run_cpmae.sh <gpu>" >&2
    exit 1
  fi
done

# ── Suite & variant lists ─────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
VARIANTS=(cpmae_frozen cpmae_ft umae_frozen umae_ft)

MAPPING_JSON="scripts/cpmae/task_mapping.json"

if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

# ── Pre-resolve per-suite metadata from task_mapping.json ─────────
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
    local suite="$1" variant="$2" job_seq="$3"

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

    # Decode variant → (encoder, ckpt, freeze)
    local encoder freeze ckpt
    case "$variant" in
        cpmae_frozen) encoder=cpmae; freeze=true;  ckpt="$CPMAE_CKPT" ;;
        cpmae_ft)     encoder=cpmae; freeze=false; ckpt="$CPMAE_CKPT" ;;
        umae_frozen)  encoder=umae;  freeze=true;  ckpt="$UMAE_CKPT"  ;;
        umae_ft)      encoder=umae;  freeze=false; ckpt="$UMAE_CKPT"  ;;
        *) echo "Unknown variant: $variant" >&2; return 1 ;;
    esac

    # Run name & dir
    local run_name="M3b_act_${suite}_${variant}"
    local run_dir="${RESULTS_DIR}/${run_name}"

    # Checkpoint skip / resume
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
    fi

    # ── Build command ─────────────────────────────────────────────
    local cmd=(
        lerobot-train
        --dataset.repo_id="$REPO_ID"
        --dataset.episodes="$episodes"
        --policy.type=act
        --policy.vision_backbone=cpmae
        --policy.cpmae_checkpoint_path="$ckpt"
        --policy.freeze_backbone="$freeze"
        --policy.num_tasks="$num_tasks"
        --policy.task_embed_dim=64
        --policy.task_index_offset="$task_index_offset"
        --env.type=libero
        --env.task="$suite"
        --env.task_ids="$env_task_ids"
        --batch_size="$BATCH_SIZE"
        --steps="$STEPS"
        --eval_freq="$EVAL_FREQ"
        --save_freq="$SAVE_FREQ"
        --eval.n_episodes="$N_EVAL_EPISODES"
        --eval.batch_size="$EVAL_BATCH"
        --seed="$SEED"
        --policy.optimizer_lr="$LR"
        --policy.optimizer_lr_backbone="$LR"
        --output_dir="$run_dir"
        --job_name="$run_name"
        --wandb.enable=true
        --wandb.project=cpmae_sweep
        --policy.push_to_hub=false
    )

    echo "[GPU ${gpu}] ${run_name} (act, ${suite}, ${variant}, ${num_tasks} tasks)"

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
export N_EVAL_EPISODES EVAL_BATCH BATCH_SIZE LR SEED
export CPMAE_CKPT UMAE_CKPT

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#SUITES[@]} * ${#VARIANTS[@]} ))
echo "=== M3b aligned sweep: 1 policy (ACT) × ${#SUITES[@]} suites × ${#VARIANTS[@]} variants = ${TOTAL_JOBS} jobs ==="
echo "Variants: ${VARIANTS[*]}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, steps: ${STEPS}, seed: ${SEED}"
echo "CP-MAE ckpt:  ${CPMAE_CKPT}"
echo "UMAE ckpt:    ${UMAE_CKPT}"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${SUITES[@]}" \
        ::: "${VARIANTS[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${SUITES[@]}" \
        ::: "${VARIANTS[@]}"
fi

echo ""
echo "========================================="
echo "M3b aligned sweep complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
