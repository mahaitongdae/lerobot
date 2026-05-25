# Shell Script Checklist for Multi-GPU Training Scripts

Apply these rules when writing or reviewing any training launcher script that runs
LIBERO (or any MuJoCo/EGL-based) environments across multiple GPUs.

---

## 1. GPU Device Assignment: EGL Probe, Not `CUDA_VISIBLE_DEVICES`

**Rule:** Never use `CUDA_VISIBLE_DEVICES` to select a GPU. Always use `RENDER_GPU_DEVICE_ID`
(resolved via `egl_probe.py`) for rendering and `--policy.device=cuda:<N>` for PyTorch.

**Why:** EGL device indices do not match CUDA device indices. Setting `CUDA_VISIBLE_DEVICES`
remaps CUDA ordinals and corrupts EGL enumeration, causing the renderer to target the wrong
GPU or fail silently.

**Required pattern:**

```bash
# ── EGL device mapping ─────────────────────────────────────────────
EGL_PROBE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/egl_probe.py"
if [[ -z "${DRY_RUN:-}" ]]; then
    EGL_MAP_JSON=$(python3 "$EGL_PROBE_SCRIPT" | grep -E '^\{.*\}$' | tail -1)
    for gpu in "${GPUS[@]}"; do
        egl_id=$(python3 -c "import json,sys; m=json.loads(sys.argv[1]); print(m.get(sys.argv[2], sys.argv[2]))" "$EGL_MAP_JSON" "$gpu")
        export "EGL_MAP_${gpu}=${egl_id}"
    done
fi
```

Inside `run_task`:

```bash
unset CUDA_VISIBLE_DEVICES
local egl_var="EGL_MAP_${gpu}"
local egl_id="${!egl_var:-$gpu}"
export RENDER_GPU_DEVICE_ID="$egl_id"
# pass explicit device to trainer:
--policy.device="cuda:${gpu}"
```

**Anti-pattern to reject:**

```bash
export CUDA_VISIBLE_DEVICES="$gpu"   # ← WRONG: corrupts EGL enumeration
```

---

## 2. Job Scheduling: `env_parallel` Slot Number `{%}`, Not Fixed GPU ID

**Rule:** Use `env_parallel` with `{%}` (slot number) to assign GPUs, not a hard-coded
index derived from a manual loop counter.

**Why:** `{%}` is the GNU Parallel *job slot* — an integer in `[1, P]` that is immediately
recycled when a job finishes. This keeps all GPU slots busy at all times. Using `{#}` (global
sequence number) or a manual modulo counter assigns GPUs correctly only in the first wave and
can drift or stall in subsequent waves.

**Required pattern:**

```bash
GPUS=("${@:-0}")
NUM_EACH_GPU=1
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

# ... define run_task, then:

export -f run_task
export GPUS RESULTS_DIR STEPS ...   # all variables used inside run_task

. env_parallel.bash

printf '%s\n' "${EXPERIMENTS[@]}" | \
    env_parallel --bar --colsep ' ' \
        --results "${RESULTS_DIR}/logs" \
        -P "${PARALLEL}" \
        run_task {1} {2} {3} {%}    # ← {%} is the slot, not {#}
```

Inside `run_task`, map slot → GPU:

```bash
run_task() {
  local ... job_slot="$4"           # receives {%}
  local device_idx=$(( (job_slot - 1) % ${#GPUS[@]} ))
  local gpu=${GPUS[$device_idx]}
  ...
}
```

**Anti-pattern to reject:**

```bash
# Manual round-robin loop — does not reuse slots dynamically:
while [ $i -lt $TOTAL ]; do
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    run_task ... $((i + g)) &
  done
  wait
  i=$((i + NUM_GPUS))
done
```

---

## Quick Checklist

Before submitting or merging a training launcher script, verify:

- [ ] `CUDA_VISIBLE_DEVICES` is **never set** (search: `grep CUDA_VISIBLE_DEVICES`)
- [ ] `egl_probe.py` is called once at startup to build `EGL_MAP_<gpu>` exports
- [ ] `RENDER_GPU_DEVICE_ID` is set inside `run_task` from `EGL_MAP_<gpu>`
- [ ] `--policy.device=cuda:<N>` is passed explicitly to the trainer
- [ ] Dispatch uses `env_parallel` with `{%}` (slot), not a manual `while`/`wait` loop
- [ ] `export -f run_task` and all referenced variables are exported before `env_parallel`
- [ ] `DRY_RUN` path uses `env_parallel` without `--bar`/`--results` (same slot logic)
