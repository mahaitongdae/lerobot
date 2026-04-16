"""Utility for loading self-supervised pretrained weights (ResNet SSL and MoCo v3 ViT)."""

import logging
import os
import re
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


# ── MoCo v3 ViT support ─────────────────────────────────────────────

_MOCOV3_KEY_MAP = {
    "cls_token": "class_token",
    "pos_embed": "encoder.pos_embedding",
    "patch_embed.proj.weight": "conv_proj.weight",
    "patch_embed.proj.bias": "conv_proj.bias",
    "norm.weight": "encoder.ln.weight",
    "norm.bias": "encoder.ln.bias",
}

_MOCOV3_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.*)")

_MOCOV3_BLOCK_KEY_MAP = {
    "norm1.weight": "ln_1.weight",
    "norm1.bias": "ln_1.bias",
    "norm2.weight": "ln_2.weight",
    "norm2.bias": "ln_2.bias",
    "attn.qkv.weight": "self_attention.in_proj_weight",
    "attn.qkv.bias": "self_attention.in_proj_bias",
    "attn.proj.weight": "self_attention.out_proj.weight",
    "attn.proj.bias": "self_attention.out_proj.bias",
    "mlp.fc1.weight": "mlp.0.weight",
    "mlp.fc1.bias": "mlp.0.bias",
    "mlp.fc2.weight": "mlp.3.weight",
    "mlp.fc2.bias": "mlp.3.bias",
}


def _remap_mocov3_to_torchvision(state_dict: dict) -> dict:
    """Remap timm-style ViT keys (MoCo v3 checkpoint) to torchvision VisionTransformer keys."""
    remapped = {}
    for key, val in state_dict.items():
        if key in _MOCOV3_KEY_MAP:
            remapped[_MOCOV3_KEY_MAP[key]] = val
            continue

        m = _MOCOV3_BLOCK_RE.match(key)
        if m:
            block_idx, suffix = m.group(1), m.group(2)
            if suffix in _MOCOV3_BLOCK_KEY_MAP:
                tv_key = f"encoder.layers.encoder_layer_{block_idx}.{_MOCOV3_BLOCK_KEY_MAP[suffix]}"
                remapped[tv_key] = val
                continue

        if any(frag in key for frag in _HEAD_KEY_FRAGMENTS):
            continue

        logger.debug("MoCo v3 key skipped (no mapping): %s", key)

    return remapped


def load_mocov3_weights_into_vit(vit_model: nn.Module, checkpoint_path: str) -> None:
    """Load MoCo v3 pretrained weights into a torchvision VisionTransformer.

    The MoCo v3 checkpoint uses timm-style key names (``blocks.0.attn.qkv.weight``)
    while torchvision uses ``encoder.layers.encoder_layer_0.self_attention.in_proj_weight``.
    This function handles the key remapping automatically.

    Args:
        vit_model: A ``torchvision.models.vision_transformer.VisionTransformer`` instance.
        checkpoint_path: Local file path or HTTP(S) URL to the MoCo v3 ``.pth.tar`` checkpoint.

    Raises:
        FileNotFoundError: If a local path does not exist.
        RuntimeError: If no backbone keys could be matched.
    """
    if checkpoint_path.startswith(("http://", "https://")):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")
    else:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MoCo v3 checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    raw_sd = _extract_state_dict(ckpt)

    # Strip the MoCo v3 wrapper prefix (module.base_encoder.)
    prefix = "module.base_encoder."
    stripped = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
    if not stripped:
        stripped = _filter_head_keys(raw_sd)

    remapped = _remap_mocov3_to_torchvision(stripped)

    target_keys = set(vit_model.state_dict().keys())
    remapped = {k: v for k, v in remapped.items() if k in target_keys}

    if len(remapped) == 0:
        sample_keys = list(raw_sd.keys())[:10]
        raise RuntimeError(
            f"Could not match any MoCo v3 checkpoint keys to the ViT backbone. "
            f"Sample checkpoint keys: {sample_keys}"
        )

    missing, unexpected = vit_model.load_state_dict(remapped, strict=False)
    logger.info(
        "Loaded MoCo v3 ViT weights from %s. matched=%d, missing=%d, unexpected=%d",
        os.path.basename(checkpoint_path) if not checkpoint_path.startswith("http") else checkpoint_path,
        len(remapped),
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.debug("Missing keys (expected for heads): %s", missing)
