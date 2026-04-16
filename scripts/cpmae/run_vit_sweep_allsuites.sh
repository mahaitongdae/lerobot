#!/usr/bin/env bash
# ViT backbone sweep: ACT with pretrained ViT encoders
# on all 4 LIBERO suites (10 tasks each), frozen encoder, lr=5e-5
#
# Backbones:
#   dinov2_vits  — DINOv2 ViT-S/14 (facebook/dinov2-small, 384-dim)
#   dinov2_vitb  — DINOv2 ViT-B/14 (facebook/dinov2-base, 768-dim)
#   siglip_vitb  — SigLIP ViT-B/16 (google/siglip-base-patch16-224, 768-dim)
#   mocov3_vits  — MoCo v3 ViT-S/16 (300ep, Facebook, 384-dim)
#
# Grid: 1 policy × 4 suites × 4 backbones = 16 jobs
# Suites: libero_10, libero_spatial, libero_object, libero_goal
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/run_vit_sweep_allsuites.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/run_vit_sweep_allsuites.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/run_vit_sweep_allsuites.sh 0                  # single GPU (sequential)
#   DRY_RUN=1 bash scripts/cpmae/run_vit_sweep_allsuites.sh 0 1      # print commands only

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
BATCH_SIZE=64
LR=5e-5
SEED=42
RESULTS_DIR="results/vit_sweep_allsuites"
REPO_ID="HuggingFaceVLA/libero"

# ── ViT backbone definitions ─────────────────────────────────────
# MoCo v3 ViT-S checkpoint URL
MOCOV3_VITS_URL="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"

# ── Suite & backbone lists ────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
BACKBONES=(dinov2_vits dinov2_vitb siglip_vitb mocov3_vits)

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

# ── Pre-download MoCo v3 checkpoint ──────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading MoCo v3 checkpoint..."
  python3 -c "
import torch, os
url = '${MOCOV3_VITS_URL}'
cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
fname = os.path.basename(url.split('?')[0])
fpath = os.path.join(cache_dir, fname)
if os.path.exists(fpath):
    print(f'  {fname} — cached')
else:
    print(f'  Downloading {fname}...')
    torch.hub.load_state_dict_from_url(url, map_location='cpu')
    print(f'  {fname} — done')
"
  echo ""

  echo "Pre-downloading HuggingFace ViT models (DINOv2, SigLIP)..."
  python3 -c "
from transformers import Dinov2Model, SiglipVisionModel
for name in ['facebook/dinov2-small', 'facebook/dinov2-base']:
    print(f'  Loading {name}...')
    Dinov2Model.from_pretrained(name)
    print(f'  {name} — cached')
print(f'  Loading google/siglip-base-patch16-224...')
SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224')
print(f'  google/siglip-base-patch16-224 — cached')
"
  echo ""
fi

# ── run_task function ──────────────────────────────────────────────
run_task() {
    local suite="$1" backbone="$2" job_seq="$3"

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

    # Run name & dir
    local run_name="ViT_act_${suite}_${backbone}"
    local run_dir="${RESULTS_DIR}/${run_name}"

    # Checkpoint skip
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
        --policy.freeze_backbone=true
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
        --wandb.project=cpmae_vit_sweep
        --policy.push_to_hub=false
    )

    # Add backbone-specific flags
    case "$backbone" in
        dinov2_vits)
            cmd+=(
                --policy.vision_backbone=dinov2
                --policy.dinov2_model_name=facebook/dinov2-small
            )
            ;;
        dinov2_vitb)
            cmd+=(
                --policy.vision_backbone=dinov2
                --policy.dinov2_model_name=facebook/dinov2-base
            )
            ;;
        siglip_vitb)
            cmd+=(
                --policy.vision_backbone=siglip
                --policy.siglip_model_name=google/siglip-base-patch16-224
            )
            ;;
        mocov3_vits)
            cmd+=(
                --policy.vision_backbone=mocov3
                --policy.mocov3_checkpoint_path="$MOCOV3_VITS_URL"
                --policy.mocov3_arch=vit_small
            )
            ;;
    esac

    echo "[GPU ${gpu}] ${run_name} (act, ${suite}, ${backbone}, ${num_tasks} tasks)"

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
export MOCOV3_VITS_URL

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#SUITES[@]} * ${#BACKBONES[@]} ))
echo "=== ViT backbone sweep: 1 policy (ACT) × ${#SUITES[@]} suites × ${#BACKBONES[@]} backbones = ${TOTAL_JOBS} jobs ==="
echo "Backbones: ${BACKBONES[*]}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, frozen encoder"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {#} \
        ::: "${SUITES[@]}" \
        ::: "${BACKBONES[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {#} \
        ::: "${SUITES[@]}" \
        ::: "${BACKBONES[@]}"
fi

echo ""
echo "========================================="
echo "ViT backbone sweep complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
