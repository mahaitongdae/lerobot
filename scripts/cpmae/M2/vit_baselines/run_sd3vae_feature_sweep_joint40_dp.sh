#!/usr/bin/env bash
# Joint 40-task SD3VAE feature sweep: Diffusion Policy only.
#
# This script keeps the policy/data setup from run_vit_sweep_joint40_dp_actl.sh,
# but sweeps which SD3 VAE encoder feature tensor is exposed to DP.
#
# Feature variants:
#   sd3vae_latent_mean     — current behavior: first latent_channels post-quant_conv channels
#   sd3vae_latent_moments  — all post-quant_conv channels, typically mean + logvar
#   sd3vae_encoder_out     — encoder output before quant_conv
#   sd3vae_mid_block       — encoder.mid_block activation
#   sd3vae_down1           — encoder.down_blocks.1 activation
#   sd3vae_down2           — encoder.down_blocks.2 activation
#   sd3vae_down3           — encoder.down_blocks.3 activation
#
# Usage:
#   bash scripts/cpmae/M2/vit_baselines/run_sd3vae_feature_sweep_joint40_dp.sh 0 1 2 3
#   DRY_RUN=1 bash scripts/cpmae/M2/vit_baselines/run_sd3vae_feature_sweep_joint40_dp.sh 0
#   BACKBONES_OVERRIDE="sd3vae_latent_mean sd3vae_down2" bash scripts/cpmae/M2/vit_baselines/run_sd3vae_feature_sweep_joint40_dp.sh 0 1

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU="${NUM_EACH_GPU:-1}"
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters ────────────────────────────────────────────────
STEPS="${STEPS:-200000}"
EVAL_FREQ="${EVAL_FREQ:-25000}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-20}"
EVAL_BATCH="${EVAL_BATCH:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-5}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-results/joint40_dp_sd3vae_features}"
REPO_ID="${REPO_ID:-HuggingFaceVLA/libero}"
WANDB_PROJECT="${WANDB_PROJECT:-cpmae_joint40_sd3vae_features}"

# ── SD3 VAE config ────────────────────────────────────────────────
SD3VAE_MODEL="${SD3VAE_MODEL:-stabilityai/stable-diffusion-3-medium-diffusers}"
SD3VAE_SUBFOLDER="${SD3VAE_SUBFOLDER:-vae}"
VAE_LATENT_PROJ_DIM="${VAE_LATENT_PROJ_DIM:-256}"
VAE_ENCODE_BATCH_SIZE="${VAE_ENCODE_BATCH_SIZE:-64}"
# Diffusion policy usually keeps native spatial maps because SpatialSoftmax handles them.
# Set this env var if an early block is too memory-heavy, e.g. SD3VAE_SPATIAL_POOL_SIZE=28.
SD3VAE_SPATIAL_POOL_SIZE="${SD3VAE_SPATIAL_POOL_SIZE:-}"

# ── Joint 40-task metadata ─────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
NUM_TASKS=40
TASK_INDEX_OFFSET=0
ENV_TASK_STR=$(IFS=,; echo "${SUITES[*]}")
MAPPING_JSON="scripts/cpmae/task_mapping.json"

if [[ -n "${BACKBONES_OVERRIDE:-}" ]]; then
  read -ra BACKBONES <<< "$BACKBONES_OVERRIDE"
else
  BACKBONES=(
    # sd3vae_latent_mean
    # sd3vae_latent_moments
    # sd3vae_encoder_out
    # sd3vae_mid_block
    sd3vae_down1
    sd3vae_down2
    sd3vae_down3
  )
fi

if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

echo "Joint 40-task SD3VAE DP feature sweep:"
echo "  Suites: ${ENV_TASK_STR}"
echo "  Total tasks: ${NUM_TASKS}"
echo "  Steps: ${STEPS}"
echo "  SD3 VAE: ${SD3VAE_MODEL}/${SD3VAE_SUBFOLDER}"
echo ""

sd3_feature_layer() {
    local backbone="$1"
    case "$backbone" in
        sd3vae|sd3vae_latent_mean)    echo "latent_mean" ;;
        sd3vae_latent_moments)        echo "latent_moments" ;;
        sd3vae_encoder_out)           echo "encoder_out" ;;
        sd3vae_mid_block)             echo "mid_block" ;;
        sd3vae_down1)                 echo "down_blocks.1" ;;
        sd3vae_down2)                 echo "down_blocks.2" ;;
        sd3vae_down3)                 echo "down_blocks.3" ;;
        *)
            echo "ERROR: unknown SD3VAE feature variant: ${backbone}" >&2
            return 1
            ;;
    esac
}
export -f sd3_feature_layer

# ── Pre-download model ────────────────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading SD3 VAE..."
  python3 - <<PYEOF
from diffusers import AutoencoderKL
print("  Loading SD3 VAE from ${SD3VAE_MODEL}...")
AutoencoderKL.from_pretrained("${SD3VAE_MODEL}", subfolder="${SD3VAE_SUBFOLDER}")
print("  SD3 VAE — cached")
PYEOF
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

run_task() {
    local backbone="$1" job_seq="$2"
    local feature_layer
    feature_layer=$(sd3_feature_layer "$backbone")

    local num_gpus=${#GPUS[@]}
    local device_idx=$(( (job_seq - 1) % num_gpus ))
    local gpu=${GPUS[$device_idx]}
    unset CUDA_VISIBLE_DEVICES
    local egl_var="EGL_MAP_${gpu}"
    local egl_id="${!egl_var:-$gpu}"
    export RENDER_GPU_DEVICE_ID="$egl_id"

    local run_name="joint40_dp_${backbone}"
    local run_dir="${RESULTS_DIR}/${run_name}"

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
            --policy.type=diffusion
            --policy.device="cuda:${gpu}"
            --policy.vision_backbone=sd3vae
            --policy.sd3vae_model_name="$SD3VAE_MODEL"
            --policy.sd3vae_subfolder="$SD3VAE_SUBFOLDER"
            --policy.sd3vae_feature_layer="$feature_layer"
            --policy.vae_latent_proj_dim="$VAE_LATENT_PROJ_DIM"
            --policy.vae_encode_batch_size="$VAE_ENCODE_BATCH_SIZE"
            --policy.freeze_backbone=false
            --policy.backbone_input_norm=identity
            --policy.num_tasks="$NUM_TASKS"
            --policy.task_embed_dim=64
            --policy.task_index_offset="$TASK_INDEX_OFFSET"
            --env.type=libero
            --env.task="$ENV_TASK_STR"
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
            --wandb.project="$WANDB_PROJECT"
            --policy.push_to_hub=false
        )

        if [[ -n "$SD3VAE_SPATIAL_POOL_SIZE" ]]; then
            cmd+=(--policy.vae_spatial_pool_size="$SD3VAE_SPATIAL_POOL_SIZE")
        fi
    fi

    echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} (diffusion, joint40, feature_layer=${feature_layer})"

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
export N_EVAL_EPISODES EVAL_BATCH BATCH_SIZE LR SEED WANDB_PROJECT
export NUM_TASKS TASK_INDEX_OFFSET ENV_TASK_STR
export SD3VAE_MODEL SD3VAE_SUBFOLDER VAE_LATENT_PROJ_DIM VAE_ENCODE_BATCH_SIZE SD3VAE_SPATIAL_POOL_SIZE

TOTAL_JOBS=${#BACKBONES[@]}
echo "=== SD3VAE feature sweep: DP × ${TOTAL_JOBS} feature variants ==="
echo "Feature variants: ${BACKBONES[*]}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, steps: ${STEPS}"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {%} \
        ::: "${BACKBONES[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        --memfree 8G \
        --memsuspend 4G \
        run_task {1} {%} \
        ::: "${BACKBONES[@]}"
fi

echo ""
echo "========================================="
echo "SD3VAE feature sweep (Diffusion Policy) complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
