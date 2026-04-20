#!/usr/bin/env bash
# Smoke test: pick a few checkpoints under results/ssl_sweep_allsuites (or RESULTS_DIR),
# run lerobot-eval in parallel, one job per GPU, no env_parallel / GNU parallel.
#
# Defaults: 4 checkpoints (first paths after sort), GPUs 0 1 2 3, 2 episodes/task.
#
# Usage (from repo root):
#   bash scripts/cpmae/smoke_eval_ssl_sweep_allsuites.sh
#   SMOKE_NUM=4 RESULTS_PREFIX=SSL_ bash scripts/cpmae/smoke_eval_ssl_sweep_allsuites.sh 0 1 2 3
#   DRY_RUN=1 bash scripts/cpmae/smoke_eval_ssl_sweep_allsuites.sh

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/ssl_sweep_allsuites}"
RESULTS_DIR="${RESULTS_DIR%/}"
RESULTS_PREFIX="${RESULTS_PREFIX:-}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-all}"
SMOKE_NUM="${SMOKE_NUM:-4}"
SMOKE_N_EVAL_EPISODES="${SMOKE_N_EVAL_EPISODES:-20}"
EVAL_LOG_SUBDIR="${EVAL_LOG_SUBDIR:-eval_logs_smoke}"

GPUS=("$@")
if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0 1 2 3)
fi

N_EVAL_EPISODES="$SMOKE_N_EVAL_EPISODES"
EVAL_BATCH_ACT=20
EVAL_BATCH_DP=20
SEED=42
REPO_ID="HuggingFaceVLA/libero"
KNOWN_SUITES="libero_10 libero_spatial libero_object libero_goal"

# ── EGL probe (same as run_eval_ssl_sweep_allsuites.sh) ───────────
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

with open('${EGL_MAP_FILE}', 'w') as f:
    json.dump(cuda_to_egl, f)
" 2>/dev/null 1>/dev/null

EGL_MAP_JSON=$(cat "${EGL_MAP_FILE}")
rm -f "${EGL_MAP_FILE}"
echo "EGL mapping (CUDA GPU -> EGL device): ${EGL_MAP_JSON}"

declare -A EGL_MAP=()
for gpu in "${GPUS[@]}"; do
    egl_id=$(python3 -c "import json; m=json.loads('${EGL_MAP_JSON}'); print(m.get('${gpu}', '${gpu}'))")
    EGL_MAP["$gpu"]="$egl_id"
    echo "  CUDA GPU ${gpu} -> EGL device ${egl_id}"
done
echo ""

# ── Discover checkpoints ─────────────────────────────────────────
if [[ -z "${RESULTS_PREFIX}" ]]; then
    run_glob="${RESULTS_DIR}/*/"
else
    run_glob="${RESULTS_DIR}/${RESULTS_PREFIX}*/"
fi

CKPT_PATHS=()
shopt -s nullglob
for run_dir in ${run_glob}; do
    [[ -d "$run_dir" ]] || continue
    case "${CHECKPOINT_MODE}" in
        all)
            for ckpt_dir in "${run_dir}"checkpoints/*/pretrained_model; do
                [[ -d "$ckpt_dir" ]] || continue
                parent=$(basename "$(dirname "$ckpt_dir")")
                [[ "$parent" == "last" ]] && continue
                CKPT_PATHS+=("$ckpt_dir")
            done
            ;;
        last)
            last_pm="${run_dir}checkpoints/last/pretrained_model"
            if [[ -d "$last_pm" ]]; then
                CKPT_PATHS+=("$last_pm")
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
                [[ -n "$best_dir" ]] && CKPT_PATHS+=("$best_dir")
            fi
            ;;
        *)
            echo "Invalid CHECKPOINT_MODE='${CHECKPOINT_MODE}' (use all or last)" >&2
            exit 1
            ;;
    esac
done
shopt -u nullglob

if [[ ${#CKPT_PATHS[@]} -eq 0 ]]; then
    echo "No checkpoints under ${RESULTS_DIR}/ (prefix=${RESULTS_PREFIX:-<all>})" >&2
    exit 1
fi

mapfile -t SMOKE_CKPTS < <(printf '%s\n' "${CKPT_PATHS[@]}" | sort -u | head -n "${SMOKE_NUM}")

echo "Smoke test: ${#SMOKE_CKPTS[@]} checkpoint(s), GPUs ${GPUS[*]}, ${N_EVAL_EPISODES} ep/task"
for p in "${SMOKE_CKPTS[@]}"; do
    echo "  - $p"
done
echo ""

LOG_DIR="${RESULTS_DIR}/${EVAL_LOG_SUBDIR}"
mkdir -p "${LOG_DIR}"

# ── run_eval: single-checkpoint invocation ───────────────────────
run_eval() {
    local ckpt_path="$1" gpu="$2" job_idx="$3"

    unset CUDA_VISIBLE_DEVICES
    local egl_id="${EGL_MAP[$gpu]:-$gpu}"
    export RENDER_GPU_DEVICE_ID="$egl_id"

    local ckpt_parent
    ckpt_parent=$(dirname "$ckpt_path")
    local step
    step=$(basename "$ckpt_parent")
    local run_dir
    run_dir=$(dirname "$(dirname "$ckpt_parent")")
    local run_name
    run_name=$(basename "$run_dir")

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

    local eval_output="${ckpt_parent}/eval_smoke"
    if [[ -n "${SMOKE_REUSE_EVAL_DIR:-}" ]]; then
        eval_output="${ckpt_parent}/eval"
    fi

    if [[ -f "${eval_output}/eval_info.json" ]]; then
        echo "[GPU ${gpu}] ${run_name}/step_${step} — already evaluated (${eval_output}), skipping"
        return 0
    fi

    local batch_size="$EVAL_BATCH_ACT"
    if [[ "$run_name" == *"_dp_"* ]]; then
        batch_size="$EVAL_BATCH_DP"
    fi

    local log_file="${LOG_DIR}/job${job_idx}_gpu${gpu}_${run_name}_step${step}.log"
    echo "[GPU ${gpu}|EGL ${egl_id}] smoke ${run_name}/step_${step} (${suite}, ${N_EVAL_EPISODES} ep/task, bs=${batch_size}) -> ${log_file}"

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
        printf '[DRY] RENDER_GPU_DEVICE_ID=%s ' "$egl_id"
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi

    "${cmd[@]}" >"${log_file}" 2>&1
}

# ── Launch: round-robin checkpoints across GPUs ──────────────────
# Enable job control so each background job runs in its own process group;
# that lets Ctrl-C kill the whole subtree (lerobot-eval + any children), not
# just the subshell wrapper.
set -m

pids=()
cleaning_up=0

cleanup() {
    [[ "$cleaning_up" -eq 1 ]] && return
    cleaning_up=1
    trap - INT TERM
    echo ""
    echo "Caught signal -> terminating ${#pids[@]} job(s)..."
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || continue
        # Negative pid = process group (job control gave each job its own pgid)
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    # Grace period, then force-kill any stragglers
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        local any_alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then any_alive=1; break; fi
        done
        [[ "$any_alive" -eq 0 ]] && break
        sleep 1
    done
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || continue
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
    echo "All jobs terminated."
    exit 130
}
trap cleanup INT TERM

job_idx=0
for ckpt in "${SMOKE_CKPTS[@]}"; do
    job_idx=$((job_idx + 1))
    gpu=${GPUS[$(( (job_idx - 1) % ${#GPUS[@]} ))]}
    ( run_eval "$ckpt" "$gpu" "$job_idx" ) &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    # `wait` returns >128 if interrupted by a signal; trap handles exit in that case.
    if ! wait "$pid"; then
        fail=$((fail + 1))
    fi
done

trap - INT TERM

echo ""
echo "Smoke eval done. ${#SMOKE_CKPTS[@]} job(s), ${fail} failure(s)."
echo "Per-job logs: ${LOG_DIR}/"
echo "Eval outputs: <checkpoint>/eval_smoke/ (or eval if SMOKE_REUSE_EVAL_DIR=1)"

exit "$fail"
