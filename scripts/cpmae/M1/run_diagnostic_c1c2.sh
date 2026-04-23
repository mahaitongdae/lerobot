#!/bin/bash
# M1 Diagnostic: Claims C1 & C2
#
# C1: Policies are more sensitive to visual feature quality at contact vs transit phases
# C2: Spatial resolution matters more than color/texture for manipulation policies
#
# Both ACT and DP use the same backbone (MoCo-v2 ResNet-50, frozen) to control
# for backbone differences — any sensitivity difference is attributable to the
# policy architecture consuming the features, not the backbone producing them.
#
# Multi-task on libero_spatial (10 tasks). HPs from M0+/M2 SSL sweep (bs64, lr5e-5).
#
# Prerequisites:
#   python scripts/cpmae/build_task_mapping.py   # generates task_mapping.json (one-time)
#   python scripts/cpmae/contact_detector.py --all_tasks --output_dir=results/contact_labels
#
# Usage:
#   bash scripts/cpmae/M1/run_diagnostic_c1c2.sh 0 1 2 3   # GPU indices
#   bash scripts/cpmae/M1/run_diagnostic_c1c2.sh 0          # single GPU
#   DRY_RUN=1 bash scripts/cpmae/M1/run_diagnostic_c1c2.sh 0  # print commands only

set -euo pipefail

GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# HP from M2 SSL sweep (same for ACT and DP — controlled comparison)
BATCH_SIZE=64
LR=5e-5
NUM_WORKERS=16

STEPS=100000
EVAL_FREQ=25000
SAVE_FREQ=100000
N_EVAL_EPISODES=20
EVAL_BATCH=20
SEED=42
RESULTS_DIR="results/M1_diagnostic_c1c2"
REPO_ID="HuggingFaceVLA/libero"
CONTACT_LABELS_DIR="results/contact_labels"
SUITE="libero_spatial"
MAPPING_JSON="scripts/cpmae/task_mapping.json"

# MoCo-v2 ResNet-50 (same backbone for both policies)
SSL_BACKBONE="moco_v2"
SSL_URL="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"

mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

# Ensure task_mapping.json exists
if [[ ! -f "$MAPPING_JSON" ]]; then
  echo "Task mapping not found. Generating..."
  python3 scripts/cpmae/build_task_mapping.py --output "$MAPPING_JSON"
fi

# Ensure contact labels exist
if [ ! -d "$CONTACT_LABELS_DIR" ] || [ -z "$(ls -A $CONTACT_LABELS_DIR 2>/dev/null)" ]; then
  echo "Generating contact labels..."
  python3 scripts/cpmae/contact_detector.py --all_tasks --output_dir="$CONTACT_LABELS_DIR"
fi

# Resolve multi-task metadata
NUM_TASKS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(m['suites']['$SUITE']['num_tasks'])")
ALL_EPISODES=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['all_episodes']) + ']')")
ENV_TASK_IDS=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print('[' + ','.join(str(e) for e in m['suites']['$SUITE']['env_task_ids']) + ']')")
TASK_INDEX_OFFSET=$(python3 -c "import json; m=json.load(open('$MAPPING_JSON')); print(min(m['suites']['$SUITE']['dataset_task_indices']))")

echo "Suite $SUITE: $NUM_TASKS tasks, offset=$TASK_INDEX_OFFSET"
echo "Backbone: $SSL_BACKBONE (same for ACT and DP)"
echo "HP: bs=$BATCH_SIZE, lr=$LR"
echo ""

# Pre-download SSL checkpoint
if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Pre-downloading MoCo-v2 checkpoint..."
  python3 -c "
import torch, os
cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
fname = os.path.basename('${SSL_URL}'.split('?')[0])
fpath = os.path.join(cache_dir, fname)
if os.path.exists(fpath):
    print(f'  {fname} — cached')
else:
    print(f'  Downloading {fname}...')
    torch.hub.load_state_dict_from_url('${SSL_URL}', map_location='cpu')
    print(f'  {fname} — done')
"
  echo ""
fi

# ── EGL device mapping ─────────────────────────────────────────────
# EGL device ordering does NOT match CUDA device ordering, and setting
# CUDA_VISIBLE_DEVICES corrupts EGL enumeration. We probe once (cached
# under .egl_probe/) and use RENDER_GPU_DEVICE_ID + --policy.device instead.
EGL_PROBE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/egl_probe.py"
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

# ── Experiments ──────────────────────────────────────────────────────
# Format: RUN_ID POLICY DEGRADE_NAME
EXPERIMENTS=(
  # M1a: Baselines (no degradation)
  "R010 act none"
  "R020 dp none"
  # M1a: Uniform degradation — C2 (resolution vs color/texture)
  "R011 act blur_sigma_2"
  "R012 act blur_sigma_5"
  "R013 act grayscale"
  "R014 act low_res_64"
  "R015 act low_res_32"
  "R021 dp blur_sigma_2"
  "R022 dp blur_sigma_5"
  "R023 dp grayscale"
  "R024 dp low_res_64"
  "R025 dp low_res_32"
  # M1b: Phase-aware degradation — C1 (contact vs transit sensitivity)
  "R030 act degrade_contact_blur5"
  "R031 act degrade_transit_blur5"
  "R032 act degrade_contact_lowres32"
  "R033 act degrade_transit_lowres32"
  "R034 dp degrade_contact_blur5"
  "R035 dp degrade_transit_blur5"
  "R036 dp degrade_contact_lowres32"
  "R037 dp degrade_transit_lowres32"
)

echo "=== M1 Diagnostic (C1+C2): ${#EXPERIMENTS[@]} experiments ==="
echo ""

# ── run_task function ──────────────────────────────────────────────
run_task() {
  local run_id="$1" policy="$2" degrade="$3" job_seq="$4"

  # GPU assignment — do NOT set CUDA_VISIBLE_DEVICES (it corrupts EGL).
  # Use RENDER_GPU_DEVICE_ID for EGL and --policy.device for PyTorch.
  local num_gpus=${#GPUS[@]}
  local device_idx=$(( (job_seq - 1) % num_gpus ))
  local gpu=${GPUS[$device_idx]}
  unset CUDA_VISIBLE_DEVICES
  local egl_var="EGL_MAP_${gpu}"
  local egl_id="${!egl_var:-$gpu}"
  export RENDER_GPU_DEVICE_ID="$egl_id"

  local run_name="${run_id}_${policy}_${degrade}"
  local run_dir="${RESULTS_DIR}/${run_name}"

  # Checkpoint skip / resume
  local resume_flag=""
  local step_file="${run_dir}/checkpoints/last/training_state/training_step.json"
  if [[ -f "$step_file" ]]; then
    local saved_step
    saved_step=$(python3 -c "import json; print(json.load(open('$step_file'))['step'])")
    if [[ "$saved_step" -ge "$STEPS" ]]; then
      echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} — completed (step ${saved_step}), skipping"
      return 0
    else
      echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} — resuming from step ${saved_step}"
      resume_flag="--resume --config_path=${run_dir}/checkpoints/last/pretrained_model/train_config.json"
    fi
  fi

  # Policy type
  local policy_type
  if [[ "$policy" == "act" ]]; then
    policy_type="act"
  else
    policy_type="diffusion"
  fi

  # Build command
  local cmd=(
    lerobot-train
    --dataset.repo_id="$REPO_ID"
    --dataset.episodes="$ALL_EPISODES"
    --policy.type="$policy_type"
    --policy.device="cuda:${gpu}"
    --policy.vision_backbone=resnet50
    --policy.freeze_backbone=true
    --policy.ssl_checkpoint_path="$SSL_URL"
    --policy.backbone_input_norm=imagenet
    --policy.num_tasks="$NUM_TASKS"
    --policy.task_embed_dim=64
    --policy.task_index_offset="$TASK_INDEX_OFFSET"
    --env.type=libero
    --env.task="$SUITE"
    --env.task_ids="$ENV_TASK_IDS"
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
    --wandb.project=cpmae_diagnostic
    --policy.push_to_hub=false
  )

  # DP needs group_norm disabled for SSL backbone
  if [[ "$policy_type" == "diffusion" ]]; then
    cmd+=(--policy.use_group_norm=false)
  fi

  # Resume flags
  if [[ -n "$resume_flag" ]]; then
    # shellcheck disable=SC2206
    cmd+=($resume_flag)
  fi

  echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name} (${policy}, ${degrade}, multi-task ${NUM_TASKS} tasks)"

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf 'RENDER_GPU_DEVICE_ID=%s ' "$egl_id"
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  if [[ "$degrade" == "none" ]]; then
    "${cmd[@]}" 2>&1 | tee "${RESULTS_DIR}/logs/${run_name}.log"
  else
    # Use degradation wrapper: replace lerobot-train with python wrapper
    # The wrapper takes --degrade_name and --contact_labels_dir before --,
    # then passes the rest to lerobot-train
    local train_args=("${cmd[@]:1}")  # strip "lerobot-train" from front
    python3 scripts/cpmae/train_with_degradation.py \
      --degrade_name="$degrade" \
      --contact_labels_dir="$CONTACT_LABELS_DIR" \
      -- "${train_args[@]}" 2>&1 | tee "${RESULTS_DIR}/logs/${run_name}.log"
  fi
}
export -f run_task

export GPUS REPO_ID RESULTS_DIR STEPS EVAL_FREQ SAVE_FREQ
export N_EVAL_EPISODES EVAL_BATCH BATCH_SIZE LR NUM_WORKERS SEED
export SSL_URL SUITE CONTACT_LABELS_DIR
export ALL_EPISODES NUM_TASKS TASK_INDEX_OFFSET ENV_TASK_IDS

# ── Launch ─────────────────────────────────────────────────────────
TOTAL=${#EXPERIMENTS[@]}
echo "Total: $TOTAL jobs on ${#GPUS[@]} GPUs"
echo "Parallel workers: ${PARALLEL}"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%s\n' "${EXPERIMENTS[@]}" | \
        env_parallel --colsep ' ' \
            -P "${PARALLEL}" \
            run_task {1} {2} {3} {%}
else
    printf '%s\n' "${EXPERIMENTS[@]}" | \
        env_parallel --bar --colsep ' ' \
            --results "${RESULTS_DIR}/logs" \
            -P "${PARALLEL}" \
            run_task {1} {2} {3} {%}
fi

echo ""
echo "========================================="
echo "M1 Diagnostic (C1+C2) Complete"
echo "========================================="
echo "Results in: $RESULTS_DIR/"
echo "Backbone: $SSL_BACKBONE (both ACT and DP)"
echo ""
echo "Next: run post-training eval:"
echo "  bash scripts/cpmae/M1/run_eval_diagnostic_c1c2.sh 0 1 2 3"
