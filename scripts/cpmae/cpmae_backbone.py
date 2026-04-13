#!/usr/bin/env python3
"""CP-MAE backbone wrapper for integration with LeRobot's ACT policy.

Loads a pretrained ViT-S encoder from CP-MAE pretraining and wraps it
to produce the same output format as ACT's other backbones (ResNet, DINOv2, SigLIP).

Output: {"feature_map": (B, C, H, W)} where H=W=14 for ViT-S/16 on 224x224 images.

Usage:
    from scripts.cpmae.cpmae_backbone import patch_act_for_cpmae
    patch_act_for_cpmae(checkpoint_path="results/M3/R200_cpmae/encoder_final.pt", freeze=True)
    # Then run lerobot-train with --policy.vision_backbone=cpmae
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CpMaeBackboneWrapper(nn.Module):
    """Wraps a pretrained CP-MAE ViT encoder for use as ACT backbone.

    Produces feature maps compatible with ACT's expected format:
    {"feature_map": (B, C, H, W)}
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        n_heads: int = 6,
        freeze: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = embed_dim  # For compatibility with ACT
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size  # 14 for 224/16

        # Build the ViT encoder (matching pretrain_mae.py architecture)
        from scripts.cpmae.pretrain_mae import ViTEncoder

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
        )

        # Load pretrained weights
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state_dict, strict=True)
            print(f"[CP-MAE] Loaded encoder from {checkpoint_path}")
        else:
            print(f"[CP-MAE] WARNING: checkpoint not found at {checkpoint_path}, using random init")

        if freeze:
            self.encoder.requires_grad_(False)
            print("[CP-MAE] Encoder frozen (no gradient updates)")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            dict with "feature_map": (B, embed_dim, grid_h, grid_w)
        """
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        encoded, _ = self.encoder(x, mask=None)  # (B, N+1, D) including CLS

        patch_tokens = encoded[:, 1:, :]  # (B, N, D)
        B, N, D = patch_tokens.shape
        h = w = int(math.sqrt(N))
        feature_map = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)

        return {"feature_map": feature_map}


def patch_act_for_cpmae(checkpoint_path: str, freeze: bool = True, **kwargs):
    """Monkey-patch ACT's model construction to support 'cpmae' vision_backbone.

    Patches:
    1. ACTConfig.__post_init__ to accept "cpmae" as a valid backbone
    2. ACT.__init__ to create CpMaeBackboneWrapper and fix the projection layer
    """
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACT

    _original_config_post_init = ACTConfig.__post_init__
    _original_act_init = ACT.__init__

    def patched_config_post_init(self):
        if self.vision_backbone == "cpmae":
            # Skip the backbone name validation for cpmae, but keep other checks
            if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
                raise NotImplementedError(
                    "`n_action_steps` must be 1 when using temporal ensembling."
                )
            if self.n_action_steps > self.chunk_size:
                raise ValueError(
                    f"The chunk size is the upper bound for n_action_steps. "
                    f"Got {self.n_action_steps} and {self.chunk_size}."
                )
            if self.n_obs_steps != 1:
                raise ValueError(f"Multiple observation steps not handled yet. Got nobs_steps={self.n_obs_steps}")
            # Call grandparent __post_init__ to skip backbone name check
            super(ACTConfig, self).__post_init__()
        else:
            _original_config_post_init(self)

    def patched_act_init(self, config, dataset_stats=None):
        if config.vision_backbone == "cpmae":
            # Save original values
            original_backbone = config.vision_backbone
            original_weights = config.pretrained_backbone_weights

            # Temporarily set to resnet18 to pass original init
            config.vision_backbone = "resnet18"
            config.pretrained_backbone_weights = None

            _original_act_init(self, config, dataset_stats)

            # Restore config
            config.vision_backbone = original_backbone
            config.pretrained_backbone_weights = original_weights

            # Replace the backbone with CP-MAE
            cpmae_wrapper = CpMaeBackboneWrapper(
                checkpoint_path=checkpoint_path,
                freeze=freeze,
                **kwargs,
            )
            self.backbone = cpmae_wrapper

            # Fix the 1x1 conv projection: was built for ResNet18 (512 channels),
            # needs to be rebuilt for CP-MAE ViT-S (384 channels).
            # ACT uses self.encoder_img_feat_input_proj
            if hasattr(self, "encoder_img_feat_input_proj"):
                device = self.encoder_img_feat_input_proj.weight.device
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    cpmae_wrapper.hidden_size, config.dim_model, kernel_size=1
                ).to(device)
                print(f"[CP-MAE] Rebuilt encoder_img_feat_input_proj: {cpmae_wrapper.hidden_size} -> {config.dim_model}")
        else:
            _original_act_init(self, config, dataset_stats)

    ACTConfig.__post_init__ = patched_config_post_init
    ACT.__init__ = patched_act_init

    print(f"[CP-MAE] Patched ACT to support vision_backbone='cpmae'")
    print(f"[CP-MAE] Checkpoint: {checkpoint_path}")
    print(f"[CP-MAE] Freeze: {freeze}")
