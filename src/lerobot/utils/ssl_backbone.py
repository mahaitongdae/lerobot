"""Utility for loading self-supervised pretrained ResNet weights (MoCo, SimCLR, BYOL, etc.)."""

import logging
import os
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Ordered by specificity — more specific prefixes first to avoid ambiguous matches.
# For downstream control policy use:
#   MoCo v1/v2: encoder_q is the gradient-trained encoder (not the momentum key encoder)
#   MoCo v3: base_encoder is the gradient-trained encoder
#   BYOL: online_encoder.net is the gradient-trained encoder (not the EMA target)
#   SimCLR / solo-learn / VISSL: backbone / encoder / trunk contain the representation
_KNOWN_PREFIXES = [
    "module.encoder_q.",
    "module.base_encoder.",
    "online_encoder.net.",
    "trunk._feature_blocks.",
    "_feature_blocks.",
    "base_model.",
    "backbone.",
    "encoder.",
    "trunk.",
    "resnet.",
    "module.",
]

# Keys belonging to projection / prediction heads that should not be loaded into the backbone.
_HEAD_KEY_FRAGMENTS = {"fc.", "projection", "predictor", "head.", "prototypes."}


def _extract_state_dict(checkpoint: dict) -> dict:
    """Unwrap the raw state_dict from common checkpoint wrapper formats.

    Handles single-level wrappers (PyTorch Lightning ``state_dict``, etc.) and
    deeply nested formats like VISSL (``classy_state_dict.base_model.model.trunk``).
    """
    # VISSL format: classy_state_dict → base_model → model → trunk (flat tensor dict)
    try:
        trunk = checkpoint["classy_state_dict"]["base_model"]["model"]["trunk"]
        if isinstance(trunk, dict) and _is_tensor_dict(trunk):
            return trunk
    except (KeyError, TypeError):
        pass

    for key in ("state_dict", "model", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _is_tensor_dict(d: dict) -> bool:
    """Return True if at least half the values are Tensors (heuristic for a flat state_dict)."""
    import torch

    if len(d) == 0:
        return False
    tensor_count = sum(1 for v in d.values() if isinstance(v, torch.Tensor))
    return tensor_count > len(d) * 0.5


def _strip_best_prefix(state_dict: dict, target_keys: set[str]) -> tuple[dict, str | None]:
    """Try each known prefix, keep the one that yields the most matches with *target_keys*."""
    # Check direct match first.
    if len(target_keys & set(state_dict.keys())) > len(target_keys) * 0.5:
        return state_dict, None

    best_prefix, best_count, best_sd = None, 0, state_dict
    for prefix in _KNOWN_PREFIXES:
        stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        match_count = len(target_keys & set(stripped.keys()))
        if match_count > best_count:
            best_count = match_count
            best_prefix = prefix
            best_sd = stripped

    return best_sd, best_prefix


def _filter_head_keys(state_dict: dict) -> dict:
    """Remove projection / prediction head parameters that are irrelevant to the backbone."""
    return {k: v for k, v in state_dict.items() if not any(frag in k for frag in _HEAD_KEY_FRAGMENTS)}


def load_ssl_weights_into_resnet(backbone_model: nn.Module, checkpoint_path: str) -> None:
    """Load SSL-pretrained weights into a torchvision ResNet model.

    Supports local paths and URLs. Automatically detects and strips key prefixes
    from MoCo v1/v2/v3, SimCLR, BYOL, VISSL, and solo-learn checkpoints.

    Args:
        backbone_model: A torchvision ResNet instance (e.g. resnet50).
        checkpoint_path: Local file path or HTTP(S) URL to the checkpoint.

    Raises:
        FileNotFoundError: If a local path is given but does not exist.
        RuntimeError: If no backbone keys could be matched after trying all known prefixes.
    """
    if checkpoint_path.startswith(("http://", "https://")):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")
    else:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SSL checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    raw_sd = _extract_state_dict(ckpt)
    target_keys = set(backbone_model.state_dict().keys())

    sd, prefix = _strip_best_prefix(raw_sd, target_keys)
    sd = _filter_head_keys(sd)

    # Keep only keys that exist in the backbone.
    sd = {k: v for k, v in sd.items() if k in target_keys}

    if len(sd) == 0:
        sample_keys = list(raw_sd.keys())[:10]
        raise RuntimeError(
            f"Could not match any SSL checkpoint keys to the ResNet backbone. "
            f"Sample checkpoint keys: {sample_keys}"
        )

    missing, unexpected = backbone_model.load_state_dict(sd, strict=False)
    logger.info(
        "Loaded SSL backbone weights from %s (prefix=%s). "
        "matched=%d, missing=%d, unexpected=%d",
        os.path.basename(checkpoint_path) if not checkpoint_path.startswith("http") else checkpoint_path,
        prefix,
        len(sd),
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.debug("Missing keys (expected for fc/head): %s", missing)
