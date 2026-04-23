#!/usr/bin/env python3
"""Probe EGL-to-CUDA device mapping and cache the result.

EGL device ordering does NOT match CUDA device ordering. This script creates
a small robosuite environment on each EGL device, measures which physical CUDA
GPU gains memory, and builds a mapping: {cuda_gpu_str: egl_device_id}.

Results are cached under .egl_probe/ in the repository root. A cached result
is reused if the GPU hardware signature (driver version + GPU list) hasn't
changed since the last probe.

Usage (standalone):
    python scripts/cpmae/egl_probe.py            # probe and print JSON
    python scripts/cpmae/egl_probe.py --force     # ignore cache, re-probe

Usage (from bash scripts):
    EGL_MAP_JSON=$(python scripts/cpmae/egl_probe.py)

Usage (from Python):
    from scripts.cpmae.egl_probe import get_egl_mapping
    mapping = get_egl_mapping()   # dict[str, int], e.g. {"0": 1, "1": 3, ...}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / ".egl_probe"


def _gpu_signature() -> str:
    """Return a stable string that changes when the GPU hardware changes."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _sig_hash(sig: str) -> str:
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def _get_gpu_mem() -> dict[int, int]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    result = {}
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        idx, mem = line.split(",")
        result[int(idx.strip())] = int(mem.strip())
    return result


def probe_egl_mapping() -> dict[str, int]:
    """Run the actual EGL probe. Returns {cuda_gpu_str: egl_device_id}."""
    import robosuite.renderers.context.egl_context as egl_mod
    from robosuite.renderers.context.egl_context import EGL

    n_egl = len(EGL.eglQueryDevicesEXT())
    print(f"Found {n_egl} EGL devices, probing...", file=sys.stderr)

    from lerobot.envs.libero import _get_suite
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = _get_suite("libero_10")
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    cuda_to_egl: dict[str, int] = {}
    for egl_id in range(n_egl):
        egl_mod.EGL_DISPLAY = None
        baseline = _get_gpu_mem()
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_heights=64, camera_widths=64,
                render_gpu_device_id=egl_id,
            )
            env.reset()
            time.sleep(0.3)
            after = _get_gpu_mem()
            diffs = {k: after[k] - baseline[k] for k in baseline if after[k] - baseline[k] > 50}
            if diffs:
                cuda_gpu = max(diffs, key=diffs.get)
                cuda_to_egl[str(cuda_gpu)] = egl_id
                print(f"  EGL {egl_id} -> CUDA GPU {cuda_gpu}", file=sys.stderr)
            else:
                print(f"  EGL {egl_id} -> no GPU memory change", file=sys.stderr)
            env.close()
            del env
            time.sleep(0.2)
        except Exception as e:
            print(f"  EGL {egl_id} -> error: {type(e).__name__}: {e}", file=sys.stderr)

    return cuda_to_egl


def get_egl_mapping(force: bool = False) -> dict[str, int]:
    """Return EGL mapping, using cache if available and hardware unchanged."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sig = _gpu_signature()
    sig_h = _sig_hash(sig)
    cache_file = CACHE_DIR / f"egl_map_{sig_h}.json"
    sig_file = CACHE_DIR / f"gpu_sig_{sig_h}.txt"

    if not force and cache_file.exists():
        mapping = json.loads(cache_file.read_text())
        print(f"Loaded cached EGL mapping from {cache_file}", file=sys.stderr)
        return mapping

    # IMPORTANT: unset CUDA_VISIBLE_DEVICES for the probe — it corrupts EGL
    # device enumeration, causing all EGL devices to collapse onto one GPU.
    saved_cvd = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        mapping = probe_egl_mapping()
    finally:
        if saved_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

    cache_file.write_text(json.dumps(mapping, indent=2))
    sig_file.write_text(sig)
    print(f"Cached EGL mapping to {cache_file}", file=sys.stderr)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Probe EGL-to-CUDA device mapping")
    parser.add_argument("--force", action="store_true", help="Ignore cache, re-probe")
    args = parser.parse_args()

    # Redirect stdout→stderr during probe to suppress library noise (e.g.
    # LIBERO's "[info] using task orders ..." printed to stdout).
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        mapping = get_egl_mapping(force=args.force)
    finally:
        sys.stdout = real_stdout

    print(json.dumps(mapping))


if __name__ == "__main__":
    main()
