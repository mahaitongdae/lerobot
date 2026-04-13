"""Inspect how many episodes belong to each LIBERO task in a dataset.

This uses the same task-to-suite matching logic as scripts/verify_task_mapping.py
and counts episodes by checking whether each task string appears in an episode's
``tasks`` field.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark

SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"] #, "libero_90"


def build_env_lookup() -> dict[str, tuple[str, int]]:
    """Map normalized task language to (suite_name, env_task_id)."""
    env_lookup = {}
    for suite_name in SUITES:
        try:
            suite = benchmark.get_benchmark_dict()[suite_name]()
        except KeyError:
            continue

        for env_task_id in range(len(suite.tasks)):
            task_language = suite.get_task(env_task_id).language.strip().lower()
            env_lookup[task_language] = (suite_name, env_task_id)

    return env_lookup


def get_task_episode_indices(meta: LeRobotDatasetMetadata, task_name: str) -> list[int]:
    return sorted(ep["episode_index"] for ep in meta.episodes if task_name in ep["tasks"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="physical-intelligence/libero")
    parser.add_argument(
        "--show-episodes",
        action="store_true",
        help="Print the episode indices for each task in addition to the count.",
    )
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.dataset_id)
    env_lookup = build_env_lookup()
    dataset_tasks = {int(row["task_index"]): name for name, row in meta.tasks.iterrows()}

    header = f"{'ds_idx':>6}  {'suite':<16}  {'env_id':>6}  {'episodes':>8}  {'dataset task'}"
    print(header)
    print("-" * max(len(header), 120))

    total_episodes = 0
    for ds_idx in sorted(dataset_tasks):
        task_name = dataset_tasks[ds_idx]
        suite_name, env_id = env_lookup.get(task_name.strip().lower(), ("???", "???"))
        episode_indices = get_task_episode_indices(meta, task_name)
        total_episodes += len(episode_indices)
        print(f"{ds_idx:>6}  {suite_name:<16}  {str(env_id):>6}  {len(episode_indices):>8}  {task_name}")
        if args.show_episodes:
            print(f"         episodes: {episode_indices}")

    print("-" * max(len(header), 120))
    print(f"{'TOTAL':>6}  {'':<16}  {'':>6}  {total_episodes:>8}  ({len(dataset_tasks)} tasks)")


if __name__ == "__main__":
    main()
