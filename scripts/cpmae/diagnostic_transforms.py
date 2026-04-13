#!/usr/bin/env python3
"""Diagnostic image degradation transforms for the visual feature study.

These transforms degrade visual inputs during training to measure policy sensitivity
to different visual properties. Used by M1 diagnostic experiments.

All transforms work on image tensors of shape (C, H, W) with values in [0, 1].

Usage in training:
    The diagnostic runner (run_diagnostic.sh) injects these transforms via a custom
    dataset wrapper that applies degradation before the policy sees the images.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform


class GaussianBlurTransform(Transform):
    """Apply Gaussian blur with a fixed sigma."""

    def __init__(self, sigma: float = 2.0, kernel_size: int | None = None):
        super().__init__()
        self.sigma = sigma
        # Kernel size should be odd and >= 6*sigma
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
        self.kernel_size = kernel_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        img = v2.functional.gaussian_blur(img, kernel_size=[self.kernel_size, self.kernel_size], sigma=[self.sigma, self.sigma])

        if squeeze:
            img = img.squeeze(0)
        return img


class GrayscaleTransform(Transform):
    """Convert to grayscale (3-channel copy)."""

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C, H, W) or (B, C, H, W)
        if img.ndim == 3:
            gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
            return gray.unsqueeze(0).expand(3, -1, -1)
        else:
            gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
            return gray.unsqueeze(1).expand(-1, 3, -1, -1)


class LowResTransform(Transform):
    """Downsample to target_size then upsample back to original size."""

    def __init__(self, target_size: int = 64):
        super().__init__()
        self.target_size = target_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        orig_h, orig_w = img.shape[-2], img.shape[-1]
        # Downsample
        img = F.interpolate(img, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        # Upsample back
        img = F.interpolate(img, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        if squeeze:
            img = img.squeeze(0)
        return img


class IdentityTransform(Transform):
    """No-op transform (baseline)."""

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img


# Registry of available degradation transforms
DEGRADATION_REGISTRY = {
    "none": lambda: IdentityTransform(),
    "blur_sigma_2": lambda: GaussianBlurTransform(sigma=2.0),
    "blur_sigma_5": lambda: GaussianBlurTransform(sigma=5.0),
    "grayscale": lambda: GrayscaleTransform(),
    "low_res_64": lambda: LowResTransform(target_size=64),
    "low_res_32": lambda: LowResTransform(target_size=32),
}


def get_degradation_transform(name: str) -> Transform:
    """Get a degradation transform by name."""
    if name not in DEGRADATION_REGISTRY:
        raise ValueError(f"Unknown degradation: {name}. Available: {list(DEGRADATION_REGISTRY.keys())}")
    return DEGRADATION_REGISTRY[name]()


class PhaseAwareDegradeTransform(Transform):
    """Apply a degradation transform only to frames of a specific phase (contact or transit).

    This requires pre-computed contact labels (from contact_detector.py).
    The transform checks the current frame's phase and applies degradation accordingly.

    Usage:
        This transform is applied within a custom dataset wrapper that provides
        the contact label for each frame alongside the image data.
    """

    def __init__(
        self,
        degrade_transform: Transform,
        target_phase: str = "contact",  # "contact" or "transit"
        contact_labels_dir: str = "results/contact_labels",
    ):
        super().__init__()
        self.degrade_transform = degrade_transform
        self.target_phase = target_phase
        self._labels_cache: dict[int, list[int]] = {}
        self.contact_labels_dir = Path(contact_labels_dir)

    def load_labels(self, task_index: int) -> dict[int, list[int]]:
        """Load contact labels for a task. Returns {episode_idx: [labels]}."""
        if task_index in self._labels_cache:
            return self._labels_cache[task_index]

        label_file = self.contact_labels_dir / f"task_{task_index:03d}_contacts.json"
        if not label_file.exists():
            raise FileNotFoundError(
                f"Contact labels not found at {label_file}. "
                f"Run: python scripts/cpmae/contact_detector.py --task_index={task_index}"
            )

        with open(label_file) as f:
            data = json.load(f)

        episode_labels = {}
        for ep_str, ep_data in data["episodes"].items():
            episode_labels[int(ep_str)] = ep_data["labels"]

        self._labels_cache[task_index] = episode_labels
        return episode_labels

    def should_degrade(self, episode_index: int, frame_index: int, task_index: int) -> bool:
        """Check if this frame should be degraded based on its phase."""
        labels = self.load_labels(task_index)
        if episode_index not in labels:
            return False

        ep_labels = labels[episode_index]
        if frame_index >= len(ep_labels):
            return False

        is_contact = ep_labels[frame_index] == 1

        if self.target_phase == "contact":
            return is_contact
        else:
            return not is_contact

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Note: In actual usage, the dataset wrapper checks should_degrade()
        # and only calls this transform when appropriate.
        return self.degrade_transform(img)


# Phase-specific degradation configs
PHASE_DEGRADATION_REGISTRY = {
    "degrade_contact_blur5": ("contact", "blur_sigma_5"),
    "degrade_transit_blur5": ("transit", "blur_sigma_5"),
    "degrade_contact_lowres32": ("contact", "low_res_32"),
    "degrade_transit_lowres32": ("transit", "low_res_32"),
}


def get_phase_degradation_config(name: str) -> tuple[str, str]:
    """Get (phase, base_degrade_name) for a phase-aware degradation."""
    if name not in PHASE_DEGRADATION_REGISTRY:
        raise ValueError(
            f"Unknown phase degradation: {name}. Available: {list(PHASE_DEGRADATION_REGISTRY.keys())}"
        )
    return PHASE_DEGRADATION_REGISTRY[name]


if __name__ == "__main__":
    # Quick test
    print("Available degradations:", list(DEGRADATION_REGISTRY.keys()))
    print("Available phase degradations:", list(PHASE_DEGRADATION_REGISTRY.keys()))

    # Test each transform
    dummy = torch.rand(3, 224, 224)
    for name in DEGRADATION_REGISTRY:
        tf = get_degradation_transform(name)
        out = tf(dummy)
        print(f"  {name}: {dummy.shape} -> {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
