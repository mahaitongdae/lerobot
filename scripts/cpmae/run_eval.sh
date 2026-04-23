#!/usr/bin/env bash
# Batch-evaluate pretrained_model checkpoints under a results tree.
#
# Discovers checkpoints under RESULTS_DIR (optional RESULTS_PREFIX filter),
# runs lerobot-eval, and saves eval_info.json alongside each checkpoint.
#
# Completed evals (eval_info.json exists) are skipped automatically.
#
# Configuration (CLI flags override env; env provides defaults):
#   --results-dir DIR   Directory containing run folders (env: RESULTS_DIR)
#   --prefix PREFIX     Only runs under ${RESULTS_DIR}/${PREFIX}* (env: RESULTS_PREFIX, empty = all subdirs)
#   --checkpoints MODE  all | last  (env: CHECKPOINT_MODE)
#                         all  — every numbered step (skips checkpoints/last duplicate)
#                         last — one checkpoint per run: checkpoints/last if present, else highest step
#
# Usage:
#   bash scripts/cpmae/run_eval_ssl_sweep_allsuites.sh --results-dir results/my_sweep --prefix SSL_ --checkpoints last 0 1
#   bash scripts/cpmae/run_eval_ssl_sweep_allsuites.sh 0 1 2 3 4 5 6 7
#   RESULTS_DIR=results/foo RESULTS_PREFIX=SSL_act_ CHECKPOINT_MODE=all bash .../run_eval_ssl_sweep_allsuites.sh 0
#   DRY_RUN=1 bash scripts/cpmae/run_eval_ssl_sweep_allsuites.sh --checkpoints last 0 1

#   RESULTS_DIR=results/foo RESULTS_PREFIX=SSL_act_ CHECKPOINT_MODE=all bash .../run_eval_ssl_sweep_allsuites.sh 0


set -euo pipefail

# ── Defaults (CLI parsing may override) ───────────────────────────
RESULTS_DIR="${RESULTS_DIR:-results/ssl_sweep_allsuites}"
RESULTS_PREFIX="${RESULTS_PREFIX:-}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-all}"

# ── Parse optional flags; remaining args are GPU ids ──────────────
GPUS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --prefix)
            RESULTS_PREFIX="$2"
            shift 2
            ;;
        --checkpoints)
            CHECKPOINT_MODE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            GPUS+=("$1")
            shift
            ;;
    esac
done
if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0)
fi

RESULTS_DIR="${RESULTS_DIR%/}"
case "${CHECKPOINT_MODE}" in
    all|last) ;;
    *)
        echo "Invalid --checkpoints / CHECKPOINT_MODE='${CHECKPOINT_MODE}' (use all or last)" >&2
        exit 1
        ;;
esac

# ── GPU & parallelism ──────────────────────────────────────────────
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ── Eval parameters ───────────────────────────────────────────────
N_EVAL_EPISODES=20       # per task (10 tasks => 200 total per checkpoint)
EVAL_BATCH_ACT=20         # batch_size for ACT (198MB model, ~25GB total fits in 32GB)
EVAL_BATCH_DP=20         # batch_size for DP  (1.1GB model, needs lower batch to fit)
SEED=42
REPO_ID="HuggingFaceVLA/libero"

# ── Known suites (for extracting suite from run name) ─────────────
KNOWN_SUITES="libero_10 libero_spatial libero_object libero_goal"

# ── Probe EGL-to-CUDA device mapping ─────────────────────────────
# EGL device ordering does NOT match CUDA device ordering.
# We probe once at startup to build cuda_gpu -> egl_device_id mapping.
echo "Probing EGL device mapping..."
EGL_MAP_FILE=$(mktemp /tmp/egl_map_XXXXXX.json)
python3 -c "
import subprocess, time, os, json, sys

def get_gpu_mem():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return {int(l.split(',')[0].strip()): int(l.split(',')[1].strip())
            for l in r.stdout.strip().split(chr(10)) if l.strip()}

import robosuite.renderers.context.egl_context as egl_mod
from robosuite.renderers.context.egl_context import EGL
n_egl = len(EGL.eglQueryDevicesEXT())

from lerobot.envs.libero import _get_suite
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

suite = _get_suite('libero_10')
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

cuda_to_egl = {}
for egl_id in range(n_egl):
    egl_mod.EGL_DISPLAY = None
    baseline = get_gpu_mem()
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64,
                                 render_gpu_device_id=egl_id)
        env.reset()
        time.sleep(0.3)
        after = get_gpu_mem()
        diffs = {k: after[k] - baseline[k] for k in baseline if after[k] - baseline[k] > 50}
        if diffs:
            cuda_gpu = max(diffs, key=diffs.get)
            cuda_to_egl[str(cuda_gpu)] = egl_id
        env.close()
        del env
        time.sleep(0.2)
    except Exception:
        pass

# Write to file to avoid stdout pollution from LIBERO/robosuite
with open('${EGL_MAP_FILE}', 'w') as f:
    json.dump(cuda_to_egl, f)
" 2>/dev/null 1>/dev/null

EGL_MAP_JSON=$(cat "${EGL_MAP_FILE}")
rm -f "${EGL_MAP_FILE}"
echo "EGL mapping (CUDA GPU -> EGL device): ${EGL_MAP_JSON}"
echo ""

# Export as flattened vars for env_parallel
for gpu in "${GPUS[@]}"; do
    egl_id=$(python3 -c "import json,sys; m=json.loads(sys.argv[1]); print(m.get(sys.argv[2], sys.argv[2]))" "$EGL_MAP_JSON" "$gpu")
    export "EGL_MAP_${gpu}=${egl_id}"
    echo "  CUDA GPU ${gpu} -> EGL device ${egl_id}"
done
echo ""

# ── Discover checkpoints ───────────────────────────────────────────
if [[ -z "${RESULTS_PREFIX}" ]]; then
    run_glob="${RESULTS_DIR}/*/"
else
    run_glob="${RESULTS_DIR}/${RESULTS_PREFIX}*/"
fi

echo "Scanning for checkpoints: RESULTS_DIR=${RESULTS_DIR} PREFIX=${RESULTS_PREFIX:-<all>} MODE=${CHECKPOINT_MODE} ..."
CKPT_PATHS=()
shopt -s nullglob

add_ckpt_if_dir() {
    local p="$1"
    [[ -d "$p" ]] || return 0
    CKPT_PATHS+=("$p")
}

for run_dir in ${run_glob}; do
    [[ -d "$run_dir" ]] || continue
    case "${CHECKPOINT_MODE}" in
        all)
            for ckpt_dir in "${run_dir}"checkpoints/*/pretrained_model; do
                [[ -d "$ckpt_dir" ]] || continue
                parent=$(basename "$(dirname "$ckpt_dir")")
                # Skip the 'last' symlink dir (same weights as a numbered step)
                [[ "$parent" == "last" ]] && continue
                add_ckpt_if_dir "$ckpt_dir"
            done
            ;;
        last)
            last_pm="${run_dir}checkpoints/last/pretrained_model"
            if [[ -d "$last_pm" ]]; then
                add_ckpt_if_dir "$last_pm"
            else
                best_dir=""
                best_n=-1
                for ckpt_dir in "${run_dir}"checkpoints/*/pretrained_model; do
                    [[ -d "$ckpt_dir" ]] || continue
                    step_name=$(basename "$(dirname "$ckpt_dir")")
                    [[ "$step_name" == "last" ]] && continue
                    if [[ "$step_name" =~ ^[0-9]+$ ]]; then
                        step_n=$((10#$step_name))
                        if [[ "$step_n" -gt "$best_n" ]]; then
                            best_n=$step_n
                            best_dir="$ckpt_dir"
                        fi
                    fi
                done
                [[ -n "$best_dir" ]] && add_ckpt_if_dir "$best_dir"
            fi
            ;;
    esac
done

shopt -u nullglob

if [[ ${#CKPT_PATHS[@]} -eq 0 ]]; then
    echo "No checkpoints found under ${RESULTS_DIR}/ (prefix=${RESULTS_PREFIX:-<all>}, mode=${CHECKPOINT_MODE})"
    exit 0
fi

echo "Found ${#CKPT_PATHS[@]} checkpoints to evaluate."
echo ""

mkdir -p "${RESULTS_DIR}/eval_logs"

# ── run_eval function ─────────────────────────────────────────────
run_eval() {
    local ckpt_path="$1" job_seq="$2"

    # GPU assignment
    local num_gpus=${#GPUS[@]}
    local device_idx=$(( (job_seq - 1) % num_gpus ))
    local gpu=${GPUS[$device_idx]}

    # IMPORTANT: Do NOT set CUDA_VISIBLE_DEVICES — it corrupts EGL device
    # enumeration (all EGL devices collapse to one GPU). Instead:
    #   - Use RENDER_GPU_DEVICE_ID for EGL rendering
    #   - Use --policy.device=cuda:$gpu for PyTorch inference
    unset CUDA_VISIBLE_DEVICES

    # Set EGL device to match the target CUDA GPU
    local egl_var="EGL_MAP_${gpu}"
    local egl_id="${!egl_var:-$gpu}"
    export RENDER_GPU_DEVICE_ID="$egl_id"

    # Parse: results/ssl_sweep_allsuites/SSL_{policy}_{suite}_{backbone}/checkpoints/{step}/pretrained_model
    local ckpt_parent
    ckpt_parent=$(dirname "$ckpt_path")            # .../checkpoints/025000
    local step
    step=$(basename "$ckpt_parent")                 # 025000
    local run_dir
    run_dir=$(dirname "$(dirname "$ckpt_parent")")  # .../SSL_act_libero_10_r3m
    local run_name
    run_name=$(basename "$run_dir")                 # SSL_act_libero_10_r3m

    # Extract suite from run name
    local suite=""
    for s in $KNOWN_SUITES; do
        if [[ "$run_name" == *"_${s}_"* ]]; then
            suite="$s"
            break
        fi
    done
    if [[ -z "$suite" ]]; then
        echo "[GPU ${gpu}] ${run_name}/step_${step} — cannot determine suite, SKIPPING"
        return 0
    fi

    # Output dir: save eval results alongside the checkpoint
    local eval_output="${ckpt_parent}/eval"

    # Skip if already evaluated
    if [[ -f "${eval_output}/eval_info.json" ]]; then
        echo "[GPU ${gpu}] ${run_name}/step_${step} — already evaluated, skipping"
        return 0
    fi

    # Select batch size by policy type (DP model is 1.1GB, needs more headroom)
    local batch_size="$EVAL_BATCH_ACT"
    if [[ "$run_name" == *"_dp_"* ]]; then
        batch_size="$EVAL_BATCH_DP"
    fi

    echo "[GPU ${gpu}|EGL ${egl_id}] ${run_name}/step_${step} (${suite}, ${N_EVAL_EPISODES} ep/task, bs=${batch_size})"

    local cmd=(
        lerobot-eval
        --policy.path="$ckpt_path"
        --policy.device="cuda:${gpu}"
        --env.type=libero
        --env.task="$suite"
        --eval.n_episodes="$N_EVAL_EPISODES"
        --eval.batch_size="$batch_size"
        --seed="$SEED"
        --dataset_repo_id="$REPO_ID"
        --output_dir="$eval_output"
    )

    if [[ -n "${DRY_RUN:-}" ]]; then
        printf 'RENDER_GPU_DEVICE_ID=%s ' "$egl_id"
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}
export -f run_eval
export GPUS N_EVAL_EPISODES EVAL_BATCH_ACT EVAL_BATCH_DP SEED REPO_ID RESULTS_DIR KNOWN_SUITES

# ── Launch ─────────────────────────────────────────────────────────
echo "=== Batch eval: ${#CKPT_PATHS[@]} checkpoints ==="
echo "RESULTS_DIR=${RESULTS_DIR}  PREFIX=${RESULTS_PREFIX:-<all>}  CHECKPOINT_MODE=${CHECKPOINT_MODE}"
echo "GPUs: ${GPUS[*]}"
echo "Parallel workers: ${PARALLEL}"
echo "Episodes per task: ${N_EVAL_EPISODES} (10 tasks = $((N_EVAL_EPISODES * 10)) total per checkpoint)"
echo ""

. env_parallel.bash

if [[ -n "${DRY_RUN:-}" ]]; then
    env_parallel -P "${PARALLEL}" \
        run_eval {1} {%} \
        ::: "${CKPT_PATHS[@]}"
else
    env_parallel --bar \
        --results "${RESULTS_DIR}/eval_logs" \
        -P "${PARALLEL}" \
        run_eval {1} {%} \
        ::: "${CKPT_PATHS[@]}"
fi

echo ""
echo "========================================="
echo "Batch evaluation complete"
echo "========================================="
echo "Results saved alongside each checkpoint as: checkpoints/<step>/eval/eval_info.json"
echo ""
