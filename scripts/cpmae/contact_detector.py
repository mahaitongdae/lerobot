#!/usr/bin/env python3
"""Contact/transit phase detector for LIBERO demonstrations.

Reads action data from the HuggingFaceVLA/libero dataset and labels each frame
as 'contact' or 'transit' based on gripper state transitions.

LIBERO actions are 7D: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
  - gripper_cmd < 0 => open (transit)
  - gripper_cmd > 0 => closed (contact / grasping)

A symmetric window is applied around each transition to capture approach/release phases.

Usage:
    python scripts/cpmae/contact_detector.py --task_index=0 --window=5
    python scripts/cpmae/contact_detector.py --all_tasks --output_dir=results/contact_labels
"""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def detect_contact_phases(
    actions: np.ndarray,
    window: int = 5,
    gripper_dim: int = -1,
    threshold: float = 0.0,
    transitions_only: bool = False,
) -> np.ndarray:
    """Label frames as contact (1) or transit (0).

    Uses gripper close/open as a proxy for contact. Known limitation:
    gripper-closed frames during carry phases are labeled as "contact"
    even though the robot is not making new contacts — this is intentional
    as these frames still require precise visual features for stable grasps.
    To restrict to transition windows only, set transitions_only=True.

    Args:
        actions: (T, action_dim) array of actions.
        window: Symmetric window (in frames) to expand around transitions.
        gripper_dim: Which action dimension is the gripper command.
        threshold: Threshold for binarizing gripper command.
        transitions_only: If True, only label frames near gripper transitions
            (not all gripper-closed frames). This gives a narrower definition
            that focuses on grasp/release events.

    Returns:
        labels: (T,) binary array. 1 = contact, 0 = transit.
    """
    T = actions.shape[0]
    gripper_cmd = actions[:, gripper_dim]

    # Binarize: positive = closed (grasping), negative = open
    is_closed = (gripper_cmd > threshold).astype(np.int32)

    # Detect transitions (open->closed or closed->open)
    transitions = np.where(np.diff(is_closed) != 0)[0]

    # Expand around transitions by window
    contact_mask = np.zeros(T, dtype=np.int32)
    for t in transitions:
        start = max(0, t - window)
        end = min(T, t + window + 1)
        contact_mask[start:end] = 1

    if not transitions_only:
        # Also mark all frames where gripper is closed as contact
        contact_mask[is_closed == 1] = 1

    return contact_mask


def get_episodes_for_task(meta: LeRobotDatasetMetadata, task_index: int) -> list[int]:
    """Get episode indices for a specific task index."""
    task_name = meta.tasks[meta.tasks["task_index"] == task_index].index[0]
    return sorted(
        ep["episode_index"] for ep in meta.episodes if task_name in ep["tasks"]
    )


def label_task_fast(
    meta: LeRobotDatasetMetadata,
    task_index: int,
    window: int = 5,
    transitions_only: bool = False,
) -> dict:
    """Fast labeling by loading only the task's episodes.

    Returns dict with per-episode labels and aggregate stats.
    """
    episode_indices = get_episodes_for_task(meta, task_index)

    # Load only the episodes we need
    dataset = LeRobotDataset(
        repo_id="HuggingFaceVLA/libero",
        episodes=episode_indices,
    )

    all_labels = {}
    total_contact = 0
    total_frames = 0

    # Iterate through the dataset frame by frame
    current_ep = None
    current_actions = []

    for i in range(len(dataset)):
        item = dataset[i]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        if ep_idx != current_ep:
            # Process previous episode
            if current_ep is not None and current_actions:
                actions = np.stack(current_actions)
                labels = detect_contact_phases(actions, window=window, transitions_only=transitions_only)
                all_labels[current_ep] = {
                    "labels": labels.tolist(),
                    "contact_ratio": float(labels.mean()),
                    "n_frames": len(labels),
                }
                total_contact += labels.sum()
                total_frames += len(labels)

            current_ep = ep_idx
            current_actions = []

        current_actions.append(item["action"].numpy())

    # Process last episode
    if current_ep is not None and current_actions:
        actions = np.stack(current_actions)
        labels = detect_contact_phases(actions, window=window, transitions_only=transitions_only)
        all_labels[current_ep] = {
            "labels": labels.tolist(),
            "contact_ratio": float(labels.mean()),
            "n_frames": len(labels),
        }
        total_contact += labels.sum()
        total_frames += len(labels)

    overall_ratio = total_contact / total_frames if total_frames > 0 else 0.0

    return {
        "task_index": task_index,
        "episodes": all_labels,
        "overall_contact_ratio": float(overall_ratio),
        "total_frames": int(total_frames),
        "total_contact_frames": int(total_contact),
        "window": window,
        "transitions_only": transitions_only,
    }


def main():
    parser = argparse.ArgumentParser(description="Contact phase detector for LIBERO")
    parser.add_argument("--task_index", type=int, default=None, help="Single task index to process")
    parser.add_argument("--all_tasks", action="store_true", help="Process all tasks")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="LIBERO suite name (for filtering tasks)")
    parser.add_argument("--window", type=int, default=5, help="Window size around transitions")
    parser.add_argument("--transitions_only", action="store_true",
                        help="Only label transition windows, not all gripper-closed frames")
    parser.add_argument("--output_dir", type=str, default="results/contact_labels", help="Output directory")
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero", help="Dataset repo ID")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(args.repo_id)

    if args.task_index is not None:
        task_indices = [args.task_index]
    elif args.all_tasks:
        task_indices = sorted(meta.tasks["task_index"].tolist())
    else:
        parser.error("Specify --task_index=N or --all_tasks")

    print(f"Processing {len(task_indices)} tasks with window={args.window}")

    for task_idx in task_indices:
        task_name = meta.tasks[meta.tasks["task_index"] == task_idx].index[0]
        print(f"\nTask {task_idx}: {task_name}")

        result = label_task_fast(meta, task_idx, window=args.window, transitions_only=args.transitions_only)

        print(f"  Total frames: {result['total_frames']}")
        print(f"  Contact frames: {result['total_contact_frames']}")
        print(f"  Contact ratio: {result['overall_contact_ratio']:.3f}")

        out_file = output_dir / f"task_{task_idx:03d}_contacts.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to: {out_file}")

    # Save aggregate summary
    if len(task_indices) > 1:
        summary_file = output_dir / "summary.json"
        summary = {"window": args.window, "tasks": {}}
        for task_idx in task_indices:
            task_file = output_dir / f"task_{task_idx:03d}_contacts.json"
            if task_file.exists():
                with open(task_file) as f:
                    data = json.load(f)
                summary["tasks"][str(task_idx)] = {
                    "contact_ratio": data["overall_contact_ratio"],
                    "total_frames": data["total_frames"],
                }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
