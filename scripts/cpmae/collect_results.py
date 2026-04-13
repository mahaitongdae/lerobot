#!/usr/bin/env python3
"""Collect and aggregate experiment results from LeRobot output directories.

Parses eval_info.json files and training logs from lerobot-train outputs,
aggregates across tasks and seeds, and produces summary JSON/CSV.

Usage:
    # Collect M0+ HP sweep results
    python scripts/cpmae/collect_results.py --input_dir=results/M0_deep --milestone=M0

    # Collect M2 baseline results
    python scripts/cpmae/collect_results.py --input_dir=results/M2_baselines --milestone=M2

    # Collect all milestones
    python scripts/cpmae/collect_results.py --input_dir=results --milestone=all --output=results/full_summary.json
"""

import argparse
import csv
import json
import re
from pathlib import Path


def find_eval_results(run_dir: Path) -> dict | None:
    """Find and parse evaluation results from a LeRobot run directory."""
    # LeRobot saves eval results in various locations
    # Check checkpoints/*/eval/ dirs
    results = {}

    # Look for eval_info.json files
    for eval_file in sorted(run_dir.rglob("eval_info.json")):
        with open(eval_file) as f:
            data = json.load(f)
        results["eval_info"] = data

    # Look for training metrics in train_info.json
    train_info = run_dir / "train_info.json"
    if train_info.exists():
        with open(train_info) as f:
            results["train_info"] = json.load(f)

    # Check for W&B summary
    wandb_dir = run_dir / "wandb"
    if wandb_dir.exists():
        for summary_file in wandb_dir.rglob("wandb-summary.json"):
            with open(summary_file) as f:
                results["wandb_summary"] = json.load(f)

    # Extract key metrics
    metrics = {}

    if "eval_info" in results:
        info = results["eval_info"]
        # LeRobot uses "overall" (not "aggregated") for top-level metrics
        overall = info.get("overall", info.get("aggregated", {}))
        if overall:
            metrics["pc_success"] = overall.get("pc_success", overall.get("avg_max_reward", None))
            metrics["avg_sum_reward"] = overall.get("avg_sum_reward")
        # Per-group results (multi-task evals)
        if "per_group" in info:
            for group_name, group_data in info["per_group"].items():
                if "pc_success" in group_data:
                    metrics.setdefault("per_group", {})[group_name] = group_data["pc_success"]
        # Fall back to per_episode
        if not metrics and "per_episode" in info:
            episodes = info["per_episode"]
            if episodes:
                # LeRobot uses "success" (bool) per episode, not "pc_success"
                successes = [e.get("success", e.get("pc_success", 0)) for e in episodes]
                metrics["pc_success"] = float(sum(1 for s in successes if s)) / len(successes) * 100
                metrics["avg_sum_reward"] = sum(e.get("sum_reward", 0) for e in episodes) / len(episodes)

    if "wandb_summary" in results:
        summary = results["wandb_summary"]
        if "eval/pc_success" in summary:
            metrics["pc_success"] = summary["eval/pc_success"]
        if "eval/avg_sum_reward" in summary:
            metrics["avg_sum_reward"] = summary["eval/avg_sum_reward"]
        if "train/loss" in summary:
            metrics["final_train_loss"] = summary["train/loss"]

    return metrics if metrics else None


def parse_run_name(run_dir: Path) -> dict:
    """Extract experiment metadata from directory name."""
    name = run_dir.name
    info = {"run_dir": str(run_dir), "run_name": name}

    # Extract task index
    task_match = re.search(r"task_?(\d+)", name)
    if task_match:
        info["task_index"] = int(task_match.group(1))

    # Extract seed
    seed_match = re.search(r"seed_?(\d+)", name)
    if seed_match:
        info["seed"] = int(seed_match.group(1))

    # Extract batch size
    bs_match = re.search(r"bs(\d+)", name)
    if bs_match:
        info["batch_size"] = int(bs_match.group(1))

    # Extract learning rate
    lr_match = re.search(r"lr([\d.e\-]+)", name)
    if lr_match:
        info["lr"] = lr_match.group(1)

    return info


def _is_leaf_run_dir(d: Path) -> bool:
    """Check if directory is a leaf-level run directory (not a parent grouping dir).

    A leaf run dir has its own self-contained training artifacts
    and no descendant that also has them.
    """
    # Must have direct evidence of a training run
    has_completion = (d / "checkpoints" / "last" / "pretrained_model").exists()
    has_config = (d / "train_config.json").exists()
    if not (has_completion or has_config):
        return False
    # Reject if any descendant also satisfies the same predicate
    for sub in d.rglob("train_config.json"):
        if sub.parent != d:
            return False
    for sub in d.rglob("checkpoints/last/pretrained_model"):
        if sub.parents[2] != d:  # checkpoints/last/pretrained_model is 3 levels
            return False
    return True


def collect_from_dir(input_dir: Path, depth: int = 4) -> list[dict]:
    """Recursively collect results from leaf run directories only."""
    all_results = []

    for run_dir in sorted(input_dir.rglob("*")):
        if not run_dir.is_dir():
            continue
        if not _is_leaf_run_dir(run_dir):
            continue

        metrics = find_eval_results(run_dir)
        run_info = parse_run_name(run_dir)

        # Derive experiment key from relative path above seed*/task_* leaves
        rel_path = run_dir.relative_to(input_dir)
        run_info["rel_path"] = str(rel_path)

        # Propagate metadata from parent dirs (seed, task, etc.) — fill in
        # keys that the leaf dir name doesn't contain
        for parent in run_dir.parents:
            if parent == input_dir:
                break
            parent_info = parse_run_name(parent)
            for k, v in parent_info.items():
                if k in ("run_dir", "run_name"):
                    continue
                # Only fill if the leaf doesn't already have this key
                run_info.setdefault(k, v)

        entry = {**run_info}
        if metrics:
            entry.update(metrics)
        else:
            entry["status"] = "no_results"

        all_results.append(entry)

    return all_results


def _experiment_key_from_path(rel_path: str) -> str:
    """Derive experiment grouping key from relative path by stripping seed/task components."""
    parts = Path(rel_path).parts
    # Keep parts that are not seed_* or task_* directories
    key_parts = [p for p in parts if not re.match(r"^(seed_?\d+|task_?\d+)$", p)]
    return "/".join(key_parts) if key_parts else rel_path


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate results across seeds and tasks."""
    from collections import defaultdict
    import numpy as np

    groups = defaultdict(list)
    for r in results:
        # Use relative path to derive experiment key (strips seed/task leaves)
        rel_path = r.get("rel_path", r.get("run_name", "unknown"))
        group_key = _experiment_key_from_path(rel_path)
        groups[group_key].append(r)

    aggregated = {}
    for group_key, group_results in groups.items():
        success_rates = [r["pc_success"] for r in group_results if "pc_success" in r]

        agg = {
            "n_runs": len(group_results),
            "n_with_results": len(success_rates),
        }

        if success_rates:
            agg["mean_pc_success"] = float(np.mean(success_rates))
            agg["std_pc_success"] = float(np.std(success_rates))
            agg["min_pc_success"] = float(np.min(success_rates))
            agg["max_pc_success"] = float(np.max(success_rates))

        # Per-task breakdown
        task_results = defaultdict(list)
        for r in group_results:
            if "task_index" in r and "pc_success" in r:
                task_results[r["task_index"]].append(r["pc_success"])

        if task_results:
            agg["per_task"] = {
                str(t): {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
                for t, vals in sorted(task_results.items())
            }

        aggregated[group_key] = agg

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to scan for results")
    parser.add_argument("--milestone", type=str, default="all",
                        help="Milestone name (M0, M1, M2, M3, M4, all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: <input_dir>/summary.json)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Also write CSV summary")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    output_file = Path(args.output) if args.output else input_dir / "summary.json"

    print(f"Scanning: {input_dir}")
    results = collect_from_dir(input_dir)
    print(f"Found {len(results)} run directories")

    # Filter by milestone if specified
    if args.milestone != "all":
        milestone_dir = input_dir
        results_with_metrics = [r for r in results if "pc_success" in r]
        results_no_metrics = [r for r in results if "pc_success" not in r]
        print(f"  With metrics: {len(results_with_metrics)}")
        print(f"  Without metrics: {len(results_no_metrics)}")

    # Aggregate
    aggregated = aggregate_results(results)

    # Build summary
    summary = {
        "input_dir": str(input_dir),
        "milestone": args.milestone,
        "total_runs": len(results),
        "runs_with_results": len([r for r in results if "pc_success" in r]),
        "aggregated": aggregated,
        "raw_results": results,
    }

    # Save JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_file}")

    # Print top-level summary
    print(f"\n{'='*60}")
    print(f"Results Summary ({args.milestone})")
    print(f"{'='*60}")
    for group_name, agg in sorted(aggregated.items()):
        if "mean_pc_success" in agg:
            print(f"  {group_name}: {agg['mean_pc_success']:.1f}% +/- {agg['std_pc_success']:.1f}% "
                  f"(n={agg['n_with_results']})")

    # Optional CSV
    if args.csv:
        csv_file = Path(args.csv)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run_name", "task_index", "seed", "pc_success", "avg_sum_reward",
                       "final_train_loss", "batch_size", "lr", "status"]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
