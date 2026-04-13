#!/usr/bin/env python3
"""Dataset wrapper that applies diagnostic image degradation to LIBERO observations.

This wraps a LeRobotDataset to apply degradation transforms to image observations
before they are passed to the policy. Supports both uniform degradation (every frame)
and phase-aware degradation (contact-only or transit-only).

Used by: scripts/cpmae/train_with_degradation.py
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from scripts.cpmae.diagnostic_transforms import (
    get_degradation_transform,
    get_phase_degradation_config,
    DEGRADATION_REGISTRY,
    PHASE_DEGRADATION_REGISTRY,
)


class DegradedDataset(Dataset):
    """Wraps a LeRobotDataset with image degradation applied to observations."""

    def __init__(
        self,
        dataset,
        degrade_name: str,
        contact_labels_dir: str | None = None,
    ):
        """
        Args:
            dataset: A LeRobotDataset instance.
            degrade_name: Name of degradation to apply. Can be:
                - A uniform degradation from DEGRADATION_REGISTRY (e.g., "blur_sigma_5")
                - A phase-aware degradation from PHASE_DEGRADATION_REGISTRY (e.g., "degrade_contact_blur5")
            contact_labels_dir: Directory containing contact label JSONs (required for phase degradations).
        """
        self.dataset = dataset
        self.degrade_name = degrade_name
        self.contact_labels_dir = Path(contact_labels_dir) if contact_labels_dir else None

        # Detect image keys from dataset metadata
        self.image_keys = list(dataset.meta.camera_keys) if hasattr(dataset.meta, "camera_keys") else []
        if not self.image_keys:
            # Fallback: look for keys starting with observation.images
            for key in dataset.meta.features:
                if key.startswith("observation.images."):
                    self.image_keys.append(key)

        # Determine if this is a phase-aware or uniform degradation
        if degrade_name in PHASE_DEGRADATION_REGISTRY:
            self.is_phase_aware = True
            phase, base_degrade = get_phase_degradation_config(degrade_name)
            self.target_phase = phase
            self.transform = get_degradation_transform(base_degrade)
            self._load_contact_labels()
        elif degrade_name in DEGRADATION_REGISTRY:
            self.is_phase_aware = False
            self.transform = get_degradation_transform(degrade_name)
        else:
            raise ValueError(
                f"Unknown degradation: {degrade_name}. "
                f"Available: {list(DEGRADATION_REGISTRY.keys()) + list(PHASE_DEGRADATION_REGISTRY.keys())}"
            )

    def _load_contact_labels(self):
        """Load all available contact labels."""
        self._contact_labels = {}  # {episode_index: [labels]}
        if self.contact_labels_dir is None:
            raise ValueError("contact_labels_dir required for phase-aware degradation")

        for label_file in sorted(self.contact_labels_dir.glob("task_*_contacts.json")):
            with open(label_file) as f:
                data = json.load(f)
            for ep_str, ep_data in data["episodes"].items():
                self._contact_labels[int(ep_str)] = ep_data["labels"]

    def _should_degrade(self, episode_index: int, frame_index: int) -> bool:
        """Check if frame should be degraded based on phase."""
        if not self.is_phase_aware:
            return True

        if episode_index not in self._contact_labels:
            return False

        labels = self._contact_labels[episode_index]
        if frame_index >= len(labels):
            return False

        is_contact = labels[frame_index] == 1
        if self.target_phase == "contact":
            return is_contact
        else:
            return not is_contact

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        episode_index = item["episode_index"].item() if torch.is_tensor(item["episode_index"]) else item["episode_index"]
        frame_index = item["frame_index"].item() if torch.is_tensor(item["frame_index"]) else item["frame_index"]

        should_degrade = self._should_degrade(episode_index, frame_index)

        if should_degrade:
            for key in self.image_keys:
                if key in item and torch.is_tensor(item[key]):
                    item[key] = self.transform(item[key])

        return item

    # Proxy all other attributes to the underlying dataset
    def __getattr__(self, name):
        if name in ("dataset", "degrade_name", "contact_labels_dir", "image_keys",
                     "is_phase_aware", "transform", "target_phase",
                     "_contact_labels", "_should_degrade"):
            raise AttributeError(name)
        return getattr(self.dataset, name)
