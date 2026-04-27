#!/usr/bin/env bash
# M5: Multi-encoder sweep — Fixed DINOv2-S on 3rd-person, varying wrist encoder
#
# 3rd-person camera (observation.images.image): DINOv2 ViT-S/14 (always)
# Wrist camera (observation.images.image2): one of 5 baselines
#
# Wrist baselines:
#   W1: siglip_vitb    — SigLIP ViT-B/16 (google/siglip-base-patch16-224)
#   W2: voltron_vcond   — Voltron V-Cond ViT-S/16 (language-conditioned, SSv2)
#   W3: vc1_vitb        — VC-1 ViT-B/16 MAE (Ego4D+ImageNet)
#   W4: cpmae_R301      — CP-MAE ViT-S/16 (contact-phase MAE, w=1.0)
#   W5: mocov3_vits     — MoCo v3 ViT-S/16 (weak baseline)
#
# Grid: 5 wrist backbones × 4 suites = 20 jobs
# Policy: Diffusion (DP), bs64, lr5e-5, frozen encoders, 100K steps
#
# Control (W0): DP + shared DINOv2-S from M2 (already complete, no rerun needed)
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/M5/run_multi_encoder_sweep.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/M5/run_multi_encoder_sweep.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/M5/run_multi_encoder_sweep.sh 0                  # single GPU
#   DRY_RUN=1 bash scripts/cpmae/M5/run_multi_encoder_sweep.sh 0 1      # print commands only

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (fixed, matching M2 DP best config) ───────────
STEPS=100000
EVAL_FREQ=25000
SAVE_FREQ=25000
N_EVAL_EPISODES=20
EVAL_BATCH=20
BATCH_SIZE=64
LR=5e-5
SEED=42
RESULTS_DIR="results/M5_multi_encoder"
REPO_ID="HuggingFaceVLA/libero"

# ── Fixed 3rd-person encoder ──────────────────────────────────────
# DINOv2 ViT-S/14 for observation.images.image (3rd-person)
THIRD_PERSON_CAM="observation.images.image"
THIRD_PERSON_BACKBONE="dinov2"
THIRD_PERSON_DINOV2_MODEL="facebook/dinov2-small"
THIRD_PERSON_NORM="imagenet"

# ── Wrist encoder checkpoints / model names ───────────────────────
WRIST_CAM="observation.images.image2"

MOCOV3_VITS_URL="${MOCOV3_VITS_URL:-https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar}"
VC1_VITB_URL="${VC1_VITB_URL:-https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth}"
VOLTRON_CACHE_DIR="${VOLTRON_CACHE_DIR:-$HOME/.voltron}"
CPMAE_R301_CKPT="${CPMAE_R301_CKPT:-results/M3_cpmae_improved/R301_cpmae/encoder_best.pt}"

# ── Suite & wrist backbone lists ──────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
if [[ -n "${BACKBONES_OVERRIDE:-}" ]]; then
  read -ra WRIST_BACKBONES <<< "$BACKBONES_OVERRIDE"
else
  WRIST_BACKBONES=(voltron_vcond vc1_vitb cpmae_R301 mocov3_vits)
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

# ── Pre-download checkpoints ──────────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../pre_download_checkpoints.sh"
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
    local suite="$1" wrist_backbone="$2" job_seq="$3"

    # GPU assignment via slot number — do NOT set CUDA_VISIBLE_DEVICES.
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
    local run_name="M5_dp_${suite}_dino3p_${wrist_backbone}_wrist"
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
        # Determine wrist-specific config flags
        local wrist_norm="imagenet"
        local -a wrist_flags=()

        case "$wrist_backbone" in
            siglip_vitb)
                wrist_norm="siglip"
                wrist_flags=(
                    "--policy.per_camera_backbone={\"${WRIST_CAM}\": \"siglip\"}"
                    "--policy.per_camera_siglip_model_name={\"${WRIST_CAM}\": \"google/siglip-base-patch16-224\"}"
                )
                ;;
            voltron_vcond)
                wrist_norm="imagenet"
                wrist_flags=(
                    "--policy.per_camera_backbone={\"${WRIST_CAM}\": \"voltron\"}"
                    "--policy.per_camera_voltron_model_id={\"${WRIST_CAM}\": \"v-cond\"}"
                    "--policy.per_camera_voltron_cache_dir={\"${WRIST_CAM}\": \"${VOLTRON_CACHE_DIR}\"}"
                )
                ;;
            vc1_vitb)
                wrist_norm="imagenet"
                wrist_flags=(
                    "--policy.per_camera_backbone={\"${WRIST_CAM}\": \"mocov3\"}"
                    "--policy.per_camera_mocov3_checkpoint_path={\"${WRIST_CAM}\": \"${VC1_VITB_URL}\"}"
                    "--policy.per_camera_mocov3_arch={\"${WRIST_CAM}\": \"vit_base\"}"
                )
                ;;
            cpmae_R301)
                wrist_norm="identity"
                wrist_flags=(
                    "--policy.per_camera_backbone={\"${WRIST_CAM}\": \"cpmae\"}"
                    "--policy.per_camera_cpmae_checkpoint_path={\"${WRIST_CAM}\": \"${CPMAE_R301_CKPT}\"}"
                    "--policy.per_camera_cpmae_embed_dim={\"${WRIST_CAM}\": 384}"
                    "--policy.per_camera_cpmae_n_heads={\"${WRIST_CAM}\": 6}"
                )
                ;;
            mocov3_vits)
                wrist_norm="imagenet"
                wrist_flags=(
                    "--policy.per_camera_backbone={\"${WRIST_CAM}\": \"mocov3\"}"
                    "--policy.per_camera_mocov3_checkpoint_path={\"${WRIST_CAM}\": \"${MOCOV3_VITS_URL}\"}"
                    "--policy.per_camera_mocov3_arch={\"${WRIST_CAM}\": \"vit_small\"}"
                )
                ;;
        esac

        cmd+=(
            --dataset.repo_id="$REPO_ID"
            --dataset.episodes="$episodes"
            --policy.type=diffusion
            --policy.device="cuda:${gpu}"
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
            --output_dir="$run_dir"
            --job_name="$run_name"
            --wandb.enable=true
            --wandb.project=cpmae_M5_multi_encoder
            --policy.push_to_hub=false
            # 3rd-person encoder (default backbone for all cameras)
            --policy.vision_backbone="$THIRD_PERSON_BACKBONE"
            --policy.dinov2_model_name="$THIRD_PERSON_DINOV2_MODEL"
            --policy.backbone_input_norm="$THIRD_PERSON_NORM"
            # Enable separate encoder per camera
            --policy.use_separate_rgb_encoder_per_camera=true
            # Wrist encoder norm
            "--policy.per_camera_backbone_norm={\"${WRIST_CAM}\": \"${wrist_norm}\"}"
        )

        # Add wrist-backbone-specific flags
        cmd+=("${wrist_flags[@]}")
    fi

    echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} (diffusion, ${suite}, wrist=${wrist_backbone})"

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
export MOCOV3_VITS_URL VC1_VITB_URL VOLTRON_CACHE_DIR CPMAE_R301_CKPT
export THIRD_PERSON_CAM THIRD_PERSON_BACKBONE THIRD_PERSON_DINOV2_MODEL THIRD_PERSON_NORM
export WRIST_CAM

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#SUITES[@]} * ${#WRIST_BACKBONES[@]} ))
echo "=== M5 Multi-encoder sweep: 4 suites × ${#WRIST_BACKBONES[@]} wrist backbones = ${TOTAL_JOBS} jobs ==="
echo "3rd-person (fixed): DINOv2-S (${THIRD_PERSON_DINOV2_MODEL})"
echo "Wrist backbones: ${WRIST_BACKBONES[*]}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, frozen encoders"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${SUITES[@]}" \
        ::: "${WRIST_BACKBONES[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {%} \
        ::: "${SUITES[@]}" \
        ::: "${WRIST_BACKBONES[@]}"
fi

echo ""
echo "========================================="
echo "M5 Multi-encoder sweep complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Control baseline (W0, shared DINOv2-S) from M2:"
echo "  goal=92.5% | spatial=92.5% | object=97.5% | libero_10=73.5% | avg=89.0%"
echo ""
