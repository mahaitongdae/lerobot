#!/usr/bin/env bash
# SSL backbone sweep: ACT and DP with different SSL-pretrained ResNet50 encoders
# on all 4 LIBERO suites (10 tasks each), frozen encoder, lr=5e-5
#
# Backbones (all ResNet-50):
#   imagenet  — supervised ImageNet (ResNet50_Weights.IMAGENET1K_V1)
#   moco_v2   — MoCo v2 800ep (Facebook)
#   moco_v1   — MoCo v1 200ep (Facebook)
#   simclr    — SimCLR 800ep (VISSL / Facebook)
#   byol      — BYOL 100ep (LightlySSL)
#   r3m       — R3M ResNet-50 (Nair et al., CoRL 2022; Ego4D time-contrastive)
#   vip       — VIP ResNet-50 (Ma et al., ICLR 2023; value-implicit pretraining)
#
# Grid: 2 policies × 4 suites × 6 backbones = 48 jobs
# Suites: libero_10, libero_spatial, libero_object, libero_goal
#
# Prerequisite for r3m baseline: `pip install git+https://github.com/facebookresearch/r3m`
# (the script auto-downloads the checkpoint via the r3m package on first run).
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#
# Usage:
#   bash scripts/cpmae/run_ssl_sweep_allsuites.sh 0 1 2 3 4 5 6 7   # 8 GPUs
#   bash scripts/cpmae/run_ssl_sweep_allsuites.sh 0 1 2 3            # 4 GPUs
#   bash scripts/cpmae/run_ssl_sweep_allsuites.sh 0                  # single GPU (sequential)
#   DRY_RUN=1 bash scripts/cpmae/run_ssl_sweep_allsuites.sh 0 1      # print commands only

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
NUM_WORKERS=16
LR=5e-5
SEED=42
RESULTS_DIR="results/ssl_sweep_allsuites"
REPO_ID="HuggingFaceVLA/libero"

# ── SSL checkpoint URLs ───────────────────────────────────────────
declare -A SSL_URLS
SSL_URLS[moco_v2]="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
SSL_URLS[moco_v1]="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar"
SSL_URLS[simclr]="https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch"
SSL_URLS[byol]="https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt"
# VIP: direct S3 URL (official PyTorch bucket).
SSL_URLS[vip]="https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
# R3M (ResNet-50) is hosted on Google Drive — pre-downloaded below via the `r3m` pip
# package into ~/.r3m/r3m_50/model.pt; SSL_URLS[r3m] is set after that download.
# "imagenet" uses supervised weights, no SSL URL needed

# ── Suite & policy lists ──────────────────────────────────────────
SUITES=(libero_10 libero_spatial libero_object libero_goal)
POLICIES=(act dp)
BACKBONES=(r3m vip imagenet moco_v2 moco_v1 simclr byol)

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

# ── Pre-download R3M checkpoint via `r3m` pip package (Google Drive) ─────────────
# R3M is only distributed via gdown; the `r3m` package handles this and caches at
# ~/.r3m/r3m_50/model.pt. We resolve that path and use it as SSL_URLS[r3m].
R3M_CKPT_PATH="$(python3 -c "import os; print(os.path.expanduser('~/.r3m/r3m_50/model.pt'))")"
if [[ -z "${DRY_RUN:-}" ]] && [[ ! -f "$R3M_CKPT_PATH" ]]; then
  echo "Pre-downloading R3M ResNet-50 via the r3m package..."
  python3 -c "
import r3m
# Triggers gdown download into ~/.r3m/r3m_50/; silently loads on CPU.
r3m.load_r3m('resnet50')
print('  R3M ResNet-50 — cached at ~/.r3m/r3m_50/model.pt')
" || {
    echo "  ERROR: failed to download R3M. Install the r3m package: pip install git+https://github.com/facebookresearch/r3m" >&2
    exit 1
  }
fi
SSL_URLS[r3m]="$R3M_CKPT_PATH"

# ── Pre-download remaining SSL checkpoints via torch.hub ──────────────────────────
# torch.hub.load_state_dict_from_url caches to ~/.cache/torch/hub/checkpoints/
# Pre-download once to avoid races when parallel jobs start simultaneously.
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading SSL checkpoints..."
  for bb in "${BACKBONES[@]}"; do
    if [[ "$bb" == "imagenet" ]] || [[ "$bb" == "r3m" ]]; then
      continue
    fi
    url="${SSL_URLS[$bb]}"
    # r3m is a local path; all others are URLs.
    if [[ "$url" == http://* ]] || [[ "$url" == https://* ]]; then
      python3 -c "
import torch, os, sys
url = '${url}'
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
    fi
  done
  echo ""
fi

# ── run_task function ──────────────────────────────────────────────
run_task() {
    local policy="$1" suite="$2" backbone="$3" job_seq="$4"

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
    local run_name="SSL_${policy}_${suite}_${backbone}"
    local run_dir="${RESULTS_DIR}/${run_name}"

    # Checkpoint skip / resume
    local resume_flag=""
    local step_file="${run_dir}/checkpoints/last/training_state/training_step.json"
    if [[ -f "$step_file" ]]; then
        local saved_step
        saved_step=$(python3 -c "import json; print(json.load(open('$step_file'))['step'])")
        if [[ "$saved_step" -ge "$STEPS" ]]; then
            echo "[GPU ${gpu}] ${run_name} — completed (step ${saved_step}), skipping"
            return 0
        else
            echo "[GPU ${gpu}] ${run_name} — resuming from step ${saved_step}"
            resume_flag="--resume --config_path=${run_dir}/checkpoints/last/pretrained_model/train_config.json"
        fi
    elif [[ -d "$run_dir" ]]; then
        echo "[GPU ${gpu}] ${run_name} — ERROR: output dir exists but no checkpoint. Remove manually: $run_dir"
        return 1
    fi

    # ── Build command ─────────────────────────────────────────────
    local cmd=()

    if [[ "$policy" == "act" ]]; then
        cmd=(
            lerobot-train
            --dataset.repo_id="$REPO_ID"
            --dataset.episodes="$episodes"
            --policy.type=act
            --policy.vision_backbone=resnet50
            --policy.freeze_backbone=true
            --policy.num_tasks="$num_tasks"
            --policy.task_embed_dim=64
            --policy.task_index_offset="$task_index_offset"
            --env.type=libero
            --env.task="$suite"
            --env.task_ids="$env_task_ids"
            --batch_size="$BATCH_SIZE"
            --num_workers="$NUM_WORKERS"
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
            --wandb.project=cpmae_ssl_sweep
            --policy.push_to_hub=false
        )
    elif [[ "$policy" == "dp" ]]; then
        cmd=(
            lerobot-train
            --dataset.repo_id="$REPO_ID"
            --dataset.episodes="$episodes"
            --policy.type=diffusion
            --policy.vision_backbone=resnet50
            --policy.freeze_backbone=true
            --policy.use_group_norm=false
            --policy.num_tasks="$num_tasks"
            --policy.task_embed_dim=64
            --policy.task_index_offset="$task_index_offset"
            --env.type=libero
            --env.task="$suite"
            --env.task_ids="$env_task_ids"
            --batch_size="$BATCH_SIZE"
            --num_workers="$NUM_WORKERS"
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
            --wandb.project=cpmae_ssl_sweep
            --policy.push_to_hub=false
        )
    fi

    # Add backbone-specific flags
    if [[ "$backbone" == "imagenet" ]]; then
        cmd+=(--policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1)
    else
        # SSL checkpoint — config auto-clears pretrained_backbone_weights and use_group_norm
        local ssl_url_var="SSL_URL_${backbone}"
        local ssl_url="${!ssl_url_var}"
        cmd+=(--policy.ssl_checkpoint_path="$ssl_url")
    fi

    # Append resume flags if resuming
    if [[ -n "$resume_flag" ]]; then
        # shellcheck disable=SC2206
        cmd+=($resume_flag)
    fi

    echo "[GPU ${gpu}] ${run_name} (${policy}, ${suite}, ${backbone}, ${num_tasks} tasks)"

    if [[ -n "${DRY_RUN:-}" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}
export -f run_task

# Export SSL URLs as individual vars (associative arrays can't be exported)
for bb in "${BACKBONES[@]}"; do
    if [[ "$bb" != "imagenet" ]]; then
        export "SSL_URL_${bb}=${SSL_URLS[$bb]}"
    fi
done
export GPUS REPO_ID RESULTS_DIR STEPS EVAL_FREQ SAVE_FREQ
export N_EVAL_EPISODES EVAL_BATCH BATCH_SIZE NUM_WORKERS LR SEED

# ── Launch ─────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#POLICIES[@]} * ${#SUITES[@]} * ${#BACKBONES[@]} ))
echo "=== SSL backbone sweep: ${#POLICIES[@]} policies × ${#SUITES[@]} suites × ${#BACKBONES[@]} backbones = ${TOTAL_JOBS} jobs ==="
echo "Backbones: ${BACKBONES[*]}"
echo "LR: ${LR}, batch_size: ${BATCH_SIZE}, frozen encoder"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_task {1} {2} {3} {%} \
        ::: "${POLICIES[@]}" ::: "${SUITES[@]}" \
        ::: "${BACKBONES[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {3} {%} \
        ::: "${POLICIES[@]}" ::: "${SUITES[@]}" \
        ::: "${BACKBONES[@]}"
fi

echo ""
echo "========================================="
echo "SSL backbone sweep complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
