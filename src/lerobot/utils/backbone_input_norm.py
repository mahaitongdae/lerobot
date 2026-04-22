#!/usr/bin/env python
"""Per-backbone pretraining input normalization.

Standard vision backbones (ResNet ImageNet, DINOv2, MoCo v3, MVP, VC-1, Voltron, SigLIP, ...)
were pretrained with specific pixel-normalization statistics. If a downstream pipeline
applies dataset-statistic normalization (e.g. LIBERO MEAN_STD) but omits the backbone's
own pretraining normalization, the backbone sees out-of-distribution inputs and its
frozen features degrade silently.

This module provides a uniform, backbone-agnostic wrapper that applies ``(x - mean)/std``
in [0, 1] pixel space just before calling the wrapped backbone. Presets cover the common
cases; a ``custom`` preset accepts arbitrary per-channel mean/std for future backbones.

Intended usage: the policy config chooses a preset via a CLI flag (e.g.
``--policy.backbone_input_norm=imagenet``) and the policy code wraps its backbone with
``BackboneInputNormalizer`` as the final construction step.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

PRESET_STATS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "identity": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "siglip": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

VALID_PRESETS: tuple[str, ...] = (*PRESET_STATS.keys(), "custom")


def resolve_preset(
    preset: str,
    mean: tuple[float, float, float] | list[float] | None = None,
    std: tuple[float, float, float] | list[float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(mean, std)`` triplets for a preset name.

    Raises ``ValueError`` on an unknown preset or on missing/invalid custom values.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"`backbone_input_norm` must be one of {VALID_PRESETS}. Got {preset!r}."
        )
    if preset == "custom":
        if mean is None or std is None:
            raise ValueError(
                "`backbone_input_norm=custom` requires both `backbone_input_mean` and "
                "`backbone_input_std` to be set (each a length-3 sequence)."
            )
        mean_t = tuple(float(v) for v in mean)
        std_t = tuple(float(v) for v in std)
        if len(mean_t) != 3 or len(std_t) != 3:
            raise ValueError(
                f"Custom mean/std must each have length 3 (RGB). Got mean={mean_t}, std={std_t}."
            )
        if any(s <= 0.0 for s in std_t):
            raise ValueError(f"Custom std must be strictly positive. Got std={std_t}.")
        return mean_t, std_t
    return PRESET_STATS[preset]


class BackboneInputNormalizer(nn.Module):
    """Wrap a vision backbone and apply ``(x - mean)/std`` to its input.

    The wrapper is transparent: it exposes the wrapped module as ``self.backbone`` and
    forwards the call with no other changes. When ``preset == "identity"`` the
    normalization becomes a no-op and the call reduces to ``self.backbone(x)``.

    Args:
        backbone: Any ``nn.Module`` that accepts an RGB image tensor of shape
            ``(B, 3, H, W)`` with values in ``[0, 1]``.
        preset: One of ``"identity" | "imagenet" | "siglip" | "custom"``.
        mean, std: Required when ``preset == "custom"``; ignored otherwise.
    """

    def __init__(
        self,
        backbone: nn.Module,
        preset: str = "identity",
        mean: tuple[float, float, float] | list[float] | None = None,
        std: tuple[float, float, float] | list[float] | None = None,
    ):
        super().__init__()
        resolved_mean, resolved_std = resolve_preset(preset, mean, std)
        self.backbone = backbone
        self.preset = preset
        self._is_identity = preset == "identity"
        self.register_buffer("mean", torch.tensor(resolved_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(resolved_std).view(1, 3, 1, 1))

        if not self._is_identity:
            logger.info(
                "BackboneInputNormalizer: preset=%s, mean=%s, std=%s",
                preset,
                resolved_mean,
                resolved_std,
            )

    def forward(self, x: Tensor):
        if not self._is_identity:
            x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)
        return self.backbone(x)
