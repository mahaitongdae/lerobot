#!/usr/bin/env bash
# Joint 40-task training: DP and ACT-L with pretrained ViT encoders
# Trains on ALL 4 LIBERO suites simultaneously (40 tasks, 1693 episodes).
#
# Key differences from per-suite scripts:
#   - All episodes from all suites combined into a single dataset
#   - num_tasks=40, task_index_offset=0 (global task indices 0-39)
#   - env.task="libero_10,libero_spatial,libero_object,libero_goal" (multi-suite eval)
#   - 4x more data → 200k steps default (2x per-suite)
#
# Backbones:
#   dinov2_vits  — DINOv2 ViT-S/14 (facebook/dinov2-small, 384-dim)
#   resnet50     — ImageNet-pretrained ResNet-50 (torchvision)
#   sd3vae       — SD3 VAE encoder (Stability AI; 16-ch latent, 8x spatial, +proj→256-dim)
#   vjepa21_vitb — V-JEPA 2.1 ViT-B/16 (Meta; 384px, 768-dim)
#
# Grid: 2 policies × N backbones = 2N jobs (one model sees all 40 tasks)
# Suites (eval): libero_10, libero_spatial, libero_object, libero_goal
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0 1 2 3 4 5 6 7
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0 1 2 3
#   POLICY=dp bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0 1
#   POLICY=actl bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0 1
#   DRY_RUN=1 bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (fixed) ────────────────────────────────────────
STEPS="${STEPS:-200000}"
EVAL_FREQ="${EVAL_FREQ:-25000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-20}"
EVAL_BATCH="${EVAL_BATCH:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-5}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-results/joint40_dp_actl}"
REPO_ID="HuggingFaceVLA/libero"

# ── Policy selection (dp, actl, or both) ──────────────────────────
POLICY="${POLICY:-both}"  # dp | actl | both

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

# ── SD3 VAE config ───────────────────────────────────────────────
SD3VAE_MODEL="${SD3VAE_MODEL:-stabilityai/stable-diffusion-3-medium-diffusers}"

# ── V-JEPA 2.1 config ────────────────────────────────────────────
VJEPA2_REPO_OR_DIR="${VJEPA2_REPO_OR_DIR:-facebookresearch/vjepa2}"
VJEPA2_MODEL="${VJEPA2_MODEL:-vjepa2_1_vit_base_384}"
VJEPA2_CHECKPOINT_URL="${VJEPA2_CHECKPOINT_URL:-}"
VJEPA2_INPUT_FRAMES="${VJEPA2_INPUT_FRAMES:-1}"
VJEPA2_ACTL_POOL_SIZE="${VJEPA2_ACTL_POOL_SIZE:-16}"

# ── Backbone list ─────────────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
if [[ -n "${BACKBONES_OVERRIDE:-}" ]]; then
  read -ra BACKBONES <<< "$BACKBONES_OVERRIDE"
else
  BACKBONES=(dinov2_vits resnet50 sd3vae vjepa21_vitb)
fi

# ── Build policy list ─────────────────────────────────────────────
case "$POLICY" in
  dp)    POLICIES=(dp) ;;
  actl)  POLICIES=(actl) ;;
  both)  POLICIES=(dp actl) ;;
  *)     echo "ERROR: POLICY must be dp, actl, or both (got: $POLICY)"; exit 1 ;;
esac

MAPPING_JSON="scripts/cpmae/task_mapping.json"

if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

# ── Resolve joint 40-task metadata from task_mapping.json ──────────
NUM_TASKS=40
TASK_INDEX_OFFSET=0
ENV_TASK_STR=$(IFS=,; echo "${SUITES[*]}")

echo "Joint 40-task training configuration:"
echo "  Suites: ${ENV_TASK_STR}"
echo "  Total tasks: ${NUM_TASKS}"
echo "  Task index offset: ${TASK_INDEX_OFFSET}"
echo "  Steps: ${STEPS}"
echo ""

uses_backbone() {
  local target="$1"
  local backbone
  for backbone in "${BACKBONES[@]}"; do
    [[ "$backbone" == "$target" ]] && return 0
  done
  return 1
}

# ── Pre-download models ──────────────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading HuggingFace models (DINOv2-S, SD3 VAE)..."
  python3 -c "
from transformers import Dinov2Model
print('  Loading facebook/dinov2-small...')
Dinov2Model.from_pretrained('facebook/dinov2-small')
print('  facebook/dinov2-small — cached')
from diffusers import AutoencoderKL
print('  Loading SD3 VAE from ${SD3VAE_MODEL}...')
AutoencoderKL.from_pretrained('${SD3VAE_MODEL}', subfolder='vae')
print('  SD3 VAE — cached')
"
  echo ""

  if uses_backbone vjepa21_vitb; then
    echo "Pre-downloading V-JEPA 2.1 backbone (${VJEPA2_MODEL})..."
    python3 - <<PYEOF
from lerobot.utils.vit_backbones import VJepa2BackboneWrapper

repo_or_dir = "${VJEPA2_REPO_OR_DIR}"
model_name = "${VJEPA2_MODEL}"
checkpoint_url = "${VJEPA2_CHECKPOINT_URL}" or None
print(f"  Loading {model_name} from {repo_or_dir}...")
encoder = VJepa2BackboneWrapper(
    repo_or_dir=repo_or_dir,
    model_name=model_name,
    checkpoint_url=checkpoint_url,
    input_frames=int("${VJEPA2_INPUT_FRAMES}"),
)
print(f"  {model_name} — cached ({encoder.hidden_size}-dim)")
PYEOF
    echo ""
  fi
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
    local policy_type="$1" backbone="$2" job_seq="$3"

    # GPU assignment
    local num_gpus=${#GPUS[@]}
    local device_idx=$(( (job_seq - 1) % num_gpus ))
    local gpu=${GPUS[$device_idx]}
    unset CUDA_VISIBLE_DEVICES
    local egl_var="EGL_MAP_${gpu}"
    local egl_id="${!egl_var:-$gpu}"
    export RENDER_GPU_DEVICE_ID="$egl_id"

    # Run name & dir
    local run_name="joint40_${policy_type}_${backbone}"
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
        # Common args for both DP and ACT-L
        cmd+=(
            --dataset.repo_id="$REPO_ID"
            --policy.device="cuda:${gpu}"
            --policy.freeze_backbone=true
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
            --wandb.project=cpmae_joint40
            --policy.push_to_hub=false
        )

        # Policy-specific args
        case "$policy_type" in
            dp)
                cmd+=(--policy.type=diffusion)
                ;;
            actl)
                cmd+=(
                    --policy.type=act
                    --policy.dim_model="$ACTL_DIM_MODEL"
                    --policy.n_heads="$ACTL_N_HEADS"
                    --policy.dim_feedforward="$ACTL_DIM_FEEDFORWARD"
                    --policy.n_encoder_layers="$ACTL_N_ENCODER_LAYERS"
                    --policy.n_decoder_layers="$ACTL_N_DECODER_LAYERS"
                    --policy.n_vae_encoder_layers="$ACTL_N_VAE_ENCODER_LAYERS"
                    --policy.latent_dim="$ACTL_LATENT_DIM"
                    --policy.use_vae=true
                    --policy.n_obs_steps="$ACTL_N_OBS_STEPS"
                    --policy.chunk_size="$ACTL_CHUNK_SIZE"
                    --policy.n_action_steps="$ACTL_N_ACTION_STEPS"
                )
                ;;
        esac

        # Per-backbone normalization
        local norm_preset="imagenet"
        case "$backbone" in
            sd3vae) norm_preset="identity" ;;
        esac
        cmd+=(--policy.backbone_input_norm="$norm_preset")

        # Per-backbone batch size overrides (ACT-L with ViT needs less memory)
        if [[ "$policy_type" == "actl" ]]; then
            case "$backbone" in
                dinov2_*)
                    cmd+=(--batch_size=16 --gradient_accumulation_steps=4)
                    ;;
                sd3vae)
                    cmd+=(--batch_size=16 --gradient_accumulation_steps=4)
                    ;;
                vjepa21_vitb)
                    cmd+=(--batch_size=16 --gradient_accumulation_steps=4)
                    ;;
            esac
        fi

        # Backbone-specific flags
        case "$backbone" in
            dinov2_vits)
                cmd+=(
                    --policy.vision_backbone=dinov2
                    --policy.dinov2_model_name=facebook/dinov2-small
                )
                ;;
            resnet50)
                cmd+=(
                    --policy.vision_backbone=resnet50
                    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1
                )
                [[ "$policy_type" == "dp" ]] && cmd+=(--policy.use_group_norm=false)
                ;;
            resnet18)
                cmd+=(
                    --policy.vision_backbone=resnet18
                    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
                )
                [[ "$policy_type" == "dp" ]] && cmd+=(--policy.use_group_norm=false)
                ;;
            sd3vae)
                # VAE encoder is frozen internally in SD3VaeBackboneWrapper;
                # freeze_backbone=false keeps the trainable 1x1 latent projection.
                cmd+=(
                    --policy.vision_backbone=sd3vae
                    --policy.sd3vae_model_name="$SD3VAE_MODEL"
                    --policy.freeze_backbone=false
                    --policy.vae_latent_proj_dim=256
                )
                # ACT flattens spatial dims to sequence tokens; pool 28x28->14x14
                # to avoid O(n²) attention OOM in the transformer encoder.
                if [[ "$policy_type" == "actl" ]]; then
                    cmd+=(--policy.vae_spatial_pool_size=14)
                fi
                ;;
            vjepa21_vitb)
                cmd+=(
                    --policy.vision_backbone=vjepa2
                    --policy.vjepa2_repo_or_dir="$VJEPA2_REPO_OR_DIR"
                    --policy.vjepa2_model_name="$VJEPA2_MODEL"
                    --policy.vjepa2_input_frames="$VJEPA2_INPUT_FRAMES"
                )
                if [[ -n "$VJEPA2_CHECKPOINT_URL" ]]; then
                    cmd+=(--policy.vjepa2_checkpoint_url="$VJEPA2_CHECKPOINT_URL")
                fi
                if [[ "$policy_type" == "actl" ]]; then
                    cmd+=(--policy.vjepa2_spatial_pool_size="$VJEPA2_ACTL_POOL_SIZE")
                fi
                ;;
        esac
    fi

    echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} (${policy_type}, joint40, ${backbone}, ${NUM_TASKS} tasks)"

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
export NUM_TASKS TASK_INDEX_OFFSET ENV_TASK_STR
export ACTL_DIM_MODEL ACTL_N_HEADS ACTL_DIM_FEEDFORWARD
export ACTL_N_ENCODER_LAYERS ACTL_N_DECODER_LAYERS ACTL_N_VAE_ENCODER_LAYERS
export ACTL_LATENT_DIM ACTL_N_OBS_STEPS ACTL_CHUNK_SIZE ACTL_N_ACTION_STEPS
export SD3VAE_MODEL VJEPA2_REPO_OR_DIR VJEPA2_MODEL VJEPA2_INPUT_FRAMES VJEPA2_ACTL_POOL_SIZE

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#POLICIES[@]} * ${#BACKBONES[@]} ))
echo "=== Joint 40-task training: ${#POLICIES[@]} policies × ${#BACKBONES[@]} backbones = ${TOTAL_JOBS} jobs ==="
echo "Policies: ${POLICIES[*]}"
echo "Backbones: ${BACKBONES[*]}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, frozen encoder, steps: ${STEPS}"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${POLICIES[@]}" \
        ::: "${BACKBONES[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        --memfree 8G \
        --memsuspend 4G \
        run_task {1} {2} {%} \
        ::: "${POLICIES[@]}" \
        ::: "${BACKBONES[@]}"
fi

echo ""
echo "========================================="
echo "Joint 40-task training complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
