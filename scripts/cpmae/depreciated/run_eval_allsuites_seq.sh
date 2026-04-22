#!/usr/bin/env bash
# Evaluate all checkpoints from the multi-task all-suites sweep
# Sequential single-GPU version (no env_parallel dependency).
#
# Discovers every pretrained_model checkpoint under RESULTS_DIR,
# runs lerobot-eval (20 episodes/task x 10 tasks = 200 episodes),
# and saves eval_info.json alongside the checkpoint.
#
# Completed evals (eval_info.json exists) are skipped automatically.
#
# Usage:
#   bash scripts/cpmae/run_eval_allsuites_seq.sh 0        # GPU 0
#   bash scripts/cpmae/run_eval_allsuites_seq.sh 3        # GPU 3
#   DRY_RUN=1 bash scripts/cpmae/run_eval_allsuites_seq.sh 0

set -euo pipefail

# ── GPU (single) ─────────────────────────────────────────────────
GPU="${1:-0}"

# ── Eval parameters ───────────────────────────────────────────────
N_EVAL_EPISODES=20
EVAL_BATCH_ACT=10
EVAL_BATCH_DP=10
SEED=42
REPO_ID="HuggingFaceVLA/libero"
RESULTS_DIR="results/multitask_allsuites"

# ── Known suites (for extracting suite from run name) ─────────────
KNOWN_SUITES="libero_10 libero_spatial libero_object libero_goal"

# ── Discover all checkpoints ──────────────────────────────────────
echo "Scanning for checkpoints in ${RESULTS_DIR}/ ..."
CKPT_PATHS=()
for run_dir in "${RESULTS_DIR}"/MT_*/; do
    [[ -d "$run_dir" ]] || continue
    for ckpt_dir in "${run_dir}"checkpoints/*/pretrained_model; do
        [[ -d "$ckpt_dir" ]] || continue
        parent=$(basename "$(dirname "$ckpt_dir")")
        [[ "$parent" == "last" ]] && continue
        CKPT_PATHS+=("$ckpt_dir")
    done
done

if [[ ${#CKPT_PATHS[@]} -eq 0 ]]; then
    echo "No checkpoints found in ${RESULTS_DIR}/"
    exit 0
fi

echo "Found ${#CKPT_PATHS[@]} checkpoints to evaluate."
echo ""

mkdir -p "${RESULTS_DIR}/eval_logs"

# ── Sequential evaluation ─────────────────────────────────────────
echo "=== Multi-task evaluation: ${#CKPT_PATHS[@]} checkpoints ==="
echo "GPU: ${GPU}"
echo "Episodes per task: ${N_EVAL_EPISODES} (10 tasks = $((N_EVAL_EPISODES * 10)) total per checkpoint)"
echo ""

export CUDA_VISIBLE_DEVICES="${GPU}"

completed=0
skipped=0
failed=0
total=${#CKPT_PATHS[@]}

for ckpt_path in "${CKPT_PATHS[@]}"; do
    ckpt_parent=$(dirname "$ckpt_path")
    step=$(basename "$ckpt_parent")
    run_dir=$(dirname "$(dirname "$ckpt_parent")")
    run_name=$(basename "$run_dir")

    # Extract suite from run name
    suite=""
    for s in $KNOWN_SUITES; do
        if [[ "$run_name" == *"_${s}_"* ]]; then
            suite="$s"
            break
        fi
    done
    if [[ -z "$suite" ]]; then
        echo "[SKIP] ${run_name}/step_${step} — cannot determine suite"
        skipped=$((skipped + 1))
        continue
    fi

    eval_output="${ckpt_parent}/eval"

    if [[ -f "${eval_output}/eval_info.json" ]]; then
        echo "[SKIP] ${run_name}/step_${step} — already evaluated"
        skipped=$((skipped + 1))
        continue
    fi

    batch_size="$EVAL_BATCH_ACT"
    if [[ "$run_name" == *"_dp_"* ]]; then
        batch_size="$EVAL_BATCH_DP"
    fi

    echo "[$(( completed + skipped + failed + 1 ))/${total}] ${run_name}/step_${step} (${suite}, ${N_EVAL_EPISODES} ep/task, bs=${batch_size})"

    cmd=(
        lerobot-eval
        --policy.path="$ckpt_path"
        --policy.device="cuda:0"
        --env.type=libero
        --env.task="$suite"
        --eval.n_episodes="$N_EVAL_EPISODES"
        --eval.batch_size="$batch_size"
        --seed="$SEED"
        --dataset_repo_id="$REPO_ID"
        --output_dir="$eval_output"
    )

    if [[ -n "${DRY_RUN:-}" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s ' "$GPU"
        printf '%q ' "${cmd[@]}"
        echo
        completed=$((completed + 1))
    else
        log_dir="${RESULTS_DIR}/eval_logs/${run_name}_step${step}"
        mkdir -p "$log_dir"
        if "${cmd[@]}" > "${log_dir}/stdout" 2> "${log_dir}/stderr"; then
            completed=$((completed + 1))
            echo "  -> done"
        else
            failed=$((failed + 1))
            echo "  -> FAILED (see ${log_dir}/stderr)"
        fi
    fi
done

echo ""
echo "========================================="
echo "Multi-task evaluation complete"
echo "========================================="
echo "  Completed: ${completed}"
echo "  Skipped:   ${skipped}"
echo "  Failed:    ${failed}"
echo "Results saved alongside each checkpoint as: checkpoints/<step>/eval/eval_info.json"
echo ""
