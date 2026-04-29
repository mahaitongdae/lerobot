#!/usr/bin/env bash
# ACT-L backbone sweep: parameter-matched ACT (ACT-L) with pretrained encoders
# on all 4 LIBERO suites (10 tasks each), frozen encoder
#
# ACT-L config: d768-enc8-dec18, n_obs_steps=2, chunk_size=16, n_action_steps=8
# Trainable params: ~257M (matches Diffusion Policy with same backbone)
#
# Backbones:
#   dinov2_vits  — DINOv2 ViT-S/14 (facebook/dinov2-small, 384-dim)
#   resnet50     — ImageNet-pretrained ResNet-50 (torchvision)
#
# Grid: 1 policy × 4 suites × 2 backbones = 8 jobs
# Suites: libero_10, libero_spatial, libero_object, libero_goal
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_actl.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_actl.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_actl.sh 0                  # single GPU
#   DRY_RUN=1 bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_actl.sh 0 1      # print only

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (fixed) ────────────────────────────────────────
STEPS=100000
EVAL_FREQ=25000
SAVE_FREQ=25000
N_EVAL_EPISODES=20
EVAL_BATCH=20
BATCH_SIZE=64
LR=5e-5
SEED=42
RESULTS_DIR="results/vit_sweep_allsuites_actl"
REPO_ID="HuggingFaceVLA/libero"

# ── ACT-L transformer config (d768-enc8-dec18, ~257M trainable) ───
ACTL_DIM_MODEL=768
ACTL_N_HEADS=12
ACTL_DIM_FEEDFORWARD=3072
ACTL_N_ENCODER_LAYERS=8
ACTL_N_DECODER_LAYERS=18
ACTL_N_VAE_ENCODER_LAYERS=4
ACTL_LATENT_DIM=32
ACTL_N_OBS_STEPS=2
ACTL_CHUNK_SIZE=16
ACTL_N_ACTION_STEPS=8

# ── Suite & backbone lists ────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
if [[ -n "${BACKBONES_OVERRIDE:-}" ]]; then
  read -ra BACKBONES <<< "$BACKBONES_OVERRIDE"
else
  BACKBONES=(dinov2_vits resnet50)
fi

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

# ── Pre-download models ──────────────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading HuggingFace ViT models (DINOv2-S)..."
  python3 -c "
from transformers import Dinov2Model
print('  Loading facebook/dinov2-small...')
Dinov2Model.from_pretrained('facebook/dinov2-small')
print('  facebook/dinov2-small — cached')
"
  echo ""
fi

# ── EGL device mapping ─────────────────────────────────────────────
EGL_PROBE_SCRIPT="scripts/cpmae/egl_probe.py"
if [[ -z "${DRY_RUN:-}" ]]; then
    echo "Loading EGL device mapping..."
    EGL_MAP_JSON=$(python3 "$EGL_PROBE_SCRIPT" | grep -E '^\{.*\}$' | tail -1)
    echo "EGL mapping (CUDA GPU -> EGL device): ${EGL_MAP_JSON}"
    for gpu in "${GPUS[@]}"; do
        egl_id=$(python3 -c "import json,sys; m=json.loads(sys.argv[1]); print(m.get(sys.argv[2], sys.argv[2]))" "$EGL_MAP_JSON" "$gpu")
        export "EGL_MAP_${gpu}=${egl_id}"
        echo "  CUDA GPU ${gpu} -> EGL device ${egl_id}"
    done
    echo ""
fi

# ── run_task function ──────────────────────────────────────────────
run_task() {
    local suite="$1" backbone="$2" job_seq="$3"

    # GPU assignment
    local num_gpus=${#GPUS[@]}
    local device_idx=$(( (job_seq - 1) % num_gpus ))
    local gpu=${GPUS[$device_idx]}
    unset CUDA_VISIBLE_DEVICES
    local egl_var="EGL_MAP_${gpu}"
    local egl_id="${!egl_var:-$gpu}"
    export RENDER_GPU_DEVICE_ID="$egl_id"

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
    local run_name="actl_${suite}_${backbone}"
    local run_dir="${RESULTS_DIR}/${run_name}"

    # Checkpoint skip / resume
    local do_resume=false
    local step_file="${run_dir}/checkpoints/last/training_state/training_step.json"
    if [[ -f "$step_file" ]]; then
        local saved_step
        saved_step=$(python3 -c "import json; print(json.load(open('$step_file'))['step'])")
        if [[ "$saved_step" -ge "$STEPS" ]]; then
            echo "[GPU ${gpu}] ${run_name} — completed (step ${saved_step}), skipping"
            return 0
        else
            echo "[GPU ${gpu}] ${run_name} — resuming from step ${saved_step}"
            do_resume=true
        fi
    elif [[ -d "$run_dir" ]]; then
        echo "[GPU ${gpu}] ${run_name} — output dir exists but no checkpoint; skipping."
        echo "  To retry, manually remove: rm -rf ${run_dir}"
        return 0
    fi

    # ── Build command ─────────────────────────────────────────────
    local cmd=(lerobot-train)

    if $do_resume; then
        local config_path="${run_dir}/checkpoints/last/pretrained_model/train_config.json"
        cmd+=(
            --resume=true
            --config_path="$config_path"
            --policy.device="cuda:${gpu}"
        )
    else
        cmd+=(
            --dataset.repo_id="$REPO_ID"
            --dataset.episodes="$episodes"
            --policy.type=act
            --policy.device="cuda:${gpu}"
            # ACT-L transformer config
            --policy.dim_model="$ACTL_DIM_MODEL"
            --policy.n_heads="$ACTL_N_HEADS"
            --policy.dim_feedforward="$ACTL_DIM_FEEDFORWARD"
            --policy.n_encoder_layers="$ACTL_N_ENCODER_LAYERS"
            --policy.n_decoder_layers="$ACTL_N_DECODER_LAYERS"
            --policy.n_vae_encoder_layers="$ACTL_N_VAE_ENCODER_LAYERS"
            --policy.latent_dim="$ACTL_LATENT_DIM"
            --policy.use_vae=true
            # Match DP observation/action structure
            --policy.n_obs_steps="$ACTL_N_OBS_STEPS"
            --policy.chunk_size="$ACTL_CHUNK_SIZE"
            --policy.n_action_steps="$ACTL_N_ACTION_STEPS"
            # Frozen backbone
            --policy.freeze_backbone=true
            # Multi-task
            --policy.num_tasks="$num_tasks"
            --policy.task_embed_dim=64
            --policy.task_index_offset="$task_index_offset"
            # Environment
            --env.type=libero
            --env.task="$suite"
            --env.task_ids="$env_task_ids"
            # Training
            --batch_size="$BATCH_SIZE"
            --steps="$STEPS"
            --eval_freq="$EVAL_FREQ"
            --save_freq="$SAVE_FREQ"
            --eval.n_episodes="$N_EVAL_EPISODES"
            --eval.batch_size="$EVAL_BATCH"
            --seed="$SEED"
            --policy.optimizer_lr="$LR"
            --output_dir="$run_dir"
            --job_name="$run_name"
            --wandb.enable=true
            --wandb.project=cpmae_vit_sweep_actl
            --policy.push_to_hub=false
        )

        # Per-backbone batch size overrides (ViT backbones produce many more
        # spatial tokens than ResNet → higher memory; halve bs and accumulate 2x).
        case "$backbone" in
            dinov2_*)
                cmd+=(--batch_size=16 --gradient_accumulation_steps=4)
                ;;
        esac

        # Per-backbone flags
        case "$backbone" in
            dinov2_vits)
                cmd+=(
                    --policy.vision_backbone=dinov2
                    --policy.dinov2_model_name=facebook/dinov2-small
                    --policy.backbone_input_norm=imagenet
                )
                ;;
            resnet50)
                cmd+=(
                    --policy.vision_backbone=resnet50
                    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1
                    --policy.backbone_input_norm=imagenet
                )
                ;;
        esac
    fi

    echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} (ACT-L, ${suite}, ${backbone}, ${num_tasks} tasks)"

    if [[ -n "${DRY_RUN:-}" ]]; then
        printf 'RENDER_GPU_DEVICE_ID=%s ' "$egl_id"
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}
export -f run_task

export GPUS REPO_ID RESULTS_DIR STEPS EVAL_FREQ SAVE_FREQ
export N_EVAL_EPISODES EVAL_BATCH BATCH_SIZE LR SEED
export ACTL_DIM_MODEL ACTL_N_HEADS ACTL_DIM_FEEDFORWARD
export ACTL_N_ENCODER_LAYERS ACTL_N_DECODER_LAYERS ACTL_N_VAE_ENCODER_LAYERS
export ACTL_LATENT_DIM ACTL_N_OBS_STEPS ACTL_CHUNK_SIZE ACTL_N_ACTION_STEPS

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#SUITES[@]} * ${#BACKBONES[@]} ))
echo "=== ACT-L backbone sweep: 1 policy × ${#SUITES[@]} suites × ${#BACKBONES[@]} backbones = ${TOTAL_JOBS} jobs ==="
echo "Backbones: ${BACKBONES[*]}"
echo "ACT-L config: d=${ACTL_DIM_MODEL}, enc=${ACTL_N_ENCODER_LAYERS}, dec=${ACTL_N_DECODER_LAYERS}, obs=${ACTL_N_OBS_STEPS}, chunk=${ACTL_CHUNK_SIZE}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, frozen encoder"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${SUITES[@]}" \
        ::: "${BACKBONES[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${SUITES[@]}" \
        ::: "${BACKBONES[@]}"
fi

echo ""
echo "========================================="
echo "ACT-L backbone sweep complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
