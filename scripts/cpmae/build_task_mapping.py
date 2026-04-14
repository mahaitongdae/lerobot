"""Build a JSON mapping between dataset task_index and LIBERO env task IDs.

Run once, then training/eval scripts read the output JSON directly.

Usage:
    python scripts/cpmae/build_task_mapping.py --dataset_id HuggingFaceVLA/libero --output scripts/cpmae/task_mapping.json

Output format:
{
  "dataset_id": "HuggingFaceVLA/libero",
  "tasks": [
    {
      "dataset_task_index": 0,
      "task_name": "pick up the black bowl...",
      "suite": "libero_10",
      "env_task_id": 4,
      "episodes": [0, 1, 2, ...]
    },
    ...
  ],
  "suites": {
    "libero_10": {
      "num_tasks": 10,
      "dataset_task_indices": [0, 1, 2, ...],
      "env_task_ids": [4, 6, ...],
      "all_episodes": [0, 1, 2, ...]
    },
    ...
  }
}
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]


def main():
    parser = argparse.ArgumentParser(description="Build dataset ↔ LIBERO env task mapping")
    parser.add_argument("--dataset_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--output", default="scripts/cpmae/task_mapping.json")
    args = parser.parse_args()

    # Suppress LIBERO's noisy prints
    import contextlib
    import logging
    import os

    logging.disable(logging.CRITICAL)
    os.environ["LIBERO_QUIET"] = "1"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from libero.libero import benchmark

        meta = LeRobotDatasetMetadata(args.dataset_id)
        bench = benchmark.get_benchmark_dict()

    logging.disable(logging.NOTSET)

    # Build env lookup: task_language -> (suite, env_task_id)
    env_lookup: dict[str, tuple[str, int]] = {}
    for suite_name in SUITES:
        if suite_name not in bench:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            suite = bench[suite_name]()
        for i in range(len(suite.tasks)):
            lang = suite.get_task(i).language.strip().lower()
            env_lookup[lang] = (suite_name, i)

    # Build per-task entries
    ds_tasks = {int(row["task_index"]): name for name, row in meta.tasks.iterrows()}
    task_entries = []
    for ds_idx in sorted(ds_tasks):
        task_name = ds_tasks[ds_idx]
        suite_name, env_id = env_lookup.get(task_name.strip().lower(), (None, None))
        episodes = sorted(ep["episode_index"] for ep in meta.episodes if task_name in ep["tasks"])
        task_entries.append({
            "dataset_task_index": ds_idx,
            "task_name": task_name,
            "suite": suite_name,
            "env_task_id": env_id,
            "episodes": episodes,
        })

    # Build per-suite summaries
    suites_summary = {}
    for suite_name in SUITES:
        suite_tasks = [t for t in task_entries if t["suite"] == suite_name]
        if not suite_tasks:
            continue
        all_eps = sorted(set(ep for t in suite_tasks for ep in t["episodes"]))
        suites_summary[suite_name] = {
            "num_tasks": len(suite_tasks),
            "dataset_task_indices": [t["dataset_task_index"] for t in suite_tasks],
            "env_task_ids": [t["env_task_id"] for t in suite_tasks],
            "all_episodes": all_eps,
        }

    output = {
        "dataset_id": args.dataset_id,
        "tasks": task_entries,
        "suites": suites_summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Wrote {out_path}")
    for suite_name, info in suites_summary.items():
        print(f"  {suite_name}: {info['num_tasks']} tasks, {len(info['all_episodes'])} episodes")


if __name__ == "__main__":
    main()
