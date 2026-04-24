#!/usr/bin/env bash
# ViT backbone sweep: Diffusion Policy with pretrained ViT encoders
# on all 4 LIBERO suites (10 tasks each), frozen encoder, lr=5e-5
#
# Backbones (smallest ViT variant per method):
#   dinov2_vits     — DINOv2 ViT-S/14 (facebook/dinov2-small, 384-dim)
#   dinov2_vitb    — DINOv2 ViT-B/14 (facebook/dinov2-base, 768-dim)
#   siglip_vitb     — SigLIP ViT-B/16 (google/siglip-base-patch16-224, 768-dim)
#   mocov3_vits     — MoCo v3 ViT-S/16 (300ep, Facebook, 384-dim)
#   mvp_vits        — MVP ViT-S/16 MAE (Xiao et al., NeurIPS 2022; ego-hoi, 384-dim)
#   vc1_vitb        — VC-1 ViT-B/16 MAE (Majumdar et al., NeurIPS 2023; ego4d+imagenet, 768-dim)
#   voltron_vcond   — Voltron V-Cond ViT-S/16 (Karamcheti et al. 2023; language-conditioned, 384-dim)
#
# Grid: 1 policy × 4 suites × 7 backbones = 28 jobs
# Suites: libero_10, libero_spatial, libero_object, libero_goal
#
# Prerequisite for voltron_vcond: `pip install voltron-robotics`
# (the script auto-downloads the Voltron checkpoint via the package on first run.)
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_dp.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_dp.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_dp.sh 0                  # single GPU (sequential)
#   DRY_RUN=1 bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_allsuites_dp.sh 0 1      # print commands only

set -euo pipefail

# ── GPU & parallelism ──────────────────────────────────────────────
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Hyperparameters (fixed) ────────────────────────────────────────
STEPS=100000
EVAL_FREQ=25000
SAVE_FREQ=100000
N_EVAL_EPISODES=20
EVAL_BATCH=20
BATCH_SIZE=64
LR=5e-5
SEED=42
RESULTS_DIR="results/vit_sweep_allsuites_dp"
REPO_ID="HuggingFaceVLA/libero"

# ── ViT backbone definitions ─────────────────────────────────────
# MoCo v3 ViT-S checkpoint URL
MOCOV3_VITS_URL="${MOCOV3_VITS_URL:-https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar}"
# MVP ViT-S MAE checkpoint (ego-hoi, 22M params, 384-dim).
MVP_VITS_URL="${MVP_VITS_URL:-https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth}"
# VC-1 ViT-B MAE checkpoint (ego4d+imagenet, 86M params, 768-dim)
VC1_VITB_URL="${VC1_VITB_URL:-https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth}"
# Voltron cache directory (the voltron-robotics package downloads to `cache/` by default)
VOLTRON_CACHE_DIR="${VOLTRON_CACHE_DIR:-$HOME/.voltron}"

# ── Suite & backbone lists ────────────────────────────────────────
SUITES=(libero_10)  #  libero_spatial libero_object libero_goal
BACKBONES=(dinov2_vits mocov3_vits vc1_vitb voltron_vcond) # dinov2_vits dinov2_vitb siglip_vitb  mvp_vits

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

# ── Pre-download ViT checkpoints ─────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading torch.hub ViT checkpoints (MoCo v3, MVP, VC-1)..."
  CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
  mkdir -p "$CACHE_DIR"

  download_and_validate() {
    local url="$1"
    local fname
    fname=$(basename "${url%%\?*}")
    local fpath="${CACHE_DIR}/${fname}"

    if [[ -f "$fpath" ]]; then
      if python3 -c "import torch, sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$fpath" >/dev/null 2>&1; then
        echo "  ${fname} — cached"
        return 0
      else
        echo "  ${fname} — cached file is corrupted, re-downloading"
        rm -f "$fpath"
      fi
    fi

    echo "  Downloading ${fname} from ${url} ..."
    if ! curl -L -f -sS --retry 3 --retry-delay 2 --connect-timeout 30 -o "${fpath}.part" "$url"; then
      rm -f "${fpath}.part"
      echo "ERROR: curl failed to download ${url}" >&2
      echo "       If this is the MVP checkpoint (Berkeley Box link), the share link" >&2
      echo "       is rate-limited or redirected. Options:" >&2
      echo "         1. Open ${url} in a browser, download manually, and place the file at" >&2
      echo "            ${fpath}" >&2
      echo "         2. Set MVP_VITS_URL=<mirror-url> and re-run this script." >&2
      exit 1
    fi

    local head4
    head4=$(head -c 4 "${fpath}.part" | od -An -c | tr -d ' \n' || true)
    local size_bytes
    size_bytes=$(stat -c%s "${fpath}.part" 2>/dev/null || stat -f%z "${fpath}.part")
    if [[ "$size_bytes" -lt 1048576 ]]; then
      rm -f "${fpath}.part"
      echo "ERROR: downloaded file is only ${size_bytes} bytes, far smaller than any" >&2
      echo "       legitimate ViT checkpoint. The URL likely served an HTML error page." >&2
      echo "       URL: ${url}" >&2
      exit 1
    fi

    if ! python3 -c "import torch, sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "${fpath}.part" >/dev/null 2>&1; then
      mv "${fpath}.part" "${fpath}.corrupt"
      echo "ERROR: downloaded file at ${fpath}.corrupt is not a valid PyTorch checkpoint." >&2
      echo "       File header bytes: ${head4}" >&2
      echo "       Size: ${size_bytes} bytes" >&2
      echo "       This is usually an HTML redirect page (Box share link failure)." >&2
      echo "       URL: ${url}" >&2
      echo "       Fix: manually download the checkpoint to ${fpath} or set MVP_VITS_URL=<mirror>." >&2
      exit 1
    fi

    mv "${fpath}.part" "$fpath"
    echo "  ${fname} — done (${size_bytes} bytes)"
  }

  for url in "$MOCOV3_VITS_URL" "$VC1_VITB_URL"; do
    download_and_validate "$url"
  done
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

  echo "Pre-downloading Voltron V-Cond checkpoint (via voltron-robotics package)..."
  mkdir -p "$VOLTRON_CACHE_DIR"
  python3 - <<PYEOF || { echo "  ERROR: failed to pre-download Voltron. Install: pip install voltron-robotics" >&2; exit 1; }
import voltron
model, _ = voltron.load('v-cond', freeze=True, cache='${VOLTRON_CACHE_DIR}')
print(f'  Voltron v-cond — cached at ${VOLTRON_CACHE_DIR}/v-cond/')
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

# ── run_task function ──────────────────────────────────────────────
run_task() {
    local suite="$1" backbone="$2" job_seq="$3"

    # GPU assignment — do NOT set CUDA_VISIBLE_DEVICES (it corrupts EGL).
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
    local run_name="ViT_dp_${suite}_${backbone}"
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
        --wandb.project=cpmae_vit_sweep_dp
        --policy.push_to_hub=false
    )

    # Per-backbone pretraining input normalization.
    local norm_preset="imagenet"
    case "$backbone" in
        siglip_*) norm_preset="siglip" ;;
        cpmae_*)  norm_preset="identity" ;;
    esac
    cmd+=(--policy.backbone_input_norm="$norm_preset")

    # DINOv2 models are larger and OOM at bs=64; halve batch size and accumulate 2x.
    case "$backbone" in
        dinov2_*)
            cmd+=(--batch_size=64) #  --gradient_accumulation_steps=4
            ;;
    esac

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
        mvp_vits)
            cmd+=(
                --policy.vision_backbone=mocov3
                --policy.mocov3_checkpoint_path="$MVP_VITS_URL"
                --policy.mocov3_arch=vit_small
            )
            ;;
        vc1_vitb)
            cmd+=(
                --policy.vision_backbone=mocov3
                --policy.mocov3_checkpoint_path="$VC1_VITB_URL"
                --policy.mocov3_arch=vit_base
            )
            ;;
        voltron_vcond)
            cmd+=(
                --policy.vision_backbone=voltron
                --policy.voltron_model_id=v-cond
                --policy.voltron_cache_dir="$VOLTRON_CACHE_DIR"
            )
            ;;
    esac

    echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} (diffusion, ${suite}, ${backbone}, ${num_tasks} tasks)"

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
export MOCOV3_VITS_URL MVP_VITS_URL VC1_VITB_URL VOLTRON_CACHE_DIR

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#SUITES[@]} * ${#BACKBONES[@]} ))
echo "=== ViT backbone sweep: 1 policy (DP) × ${#SUITES[@]} suites × ${#BACKBONES[@]} backbones = ${TOTAL_JOBS} jobs ==="
echo "Backbones: ${BACKBONES[*]}"
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
echo "ViT backbone sweep (Diffusion Policy) complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
