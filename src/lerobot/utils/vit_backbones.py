"""Shared ViT backbone wrappers for policy image encoders.

Each wrapper produces a dict ``{"feature_map": (B, C, H, W)}`` from a raw image
tensor ``(B, 3, H_in, W_in)``, matching the interface used by ResNet backbones
(via ``IntermediateLayerGetter``). This allows ``SpatialSoftmax`` and other
downstream modules to treat ViT and ResNet features identically.
"""

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from transformers import Dinov2Model, SiglipVisionModel
except ImportError:
    Dinov2Model = None
    SiglipVisionModel = None

from lerobot.utils.ssl_backbone import load_mocov3_weights_into_vit


class SiglipBackboneWrapper(nn.Module):
    """Wraps a SigLIP vision model to produce 2D feature maps.

    SigLIP outputs (B, num_patches, hidden_size) patch tokens. This wrapper reshapes them
    to (B, hidden_size, H, W) and returns {"feature_map": ...} to match the ResNet backbone interface.
    """

    def __init__(self, model_name: str):
        super().__init__()
        if SiglipVisionModel is None:
            raise ImportError(
                "SigLIP backbone requires the `transformers` library. "
                "Install it with: pip install transformers"
            )
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size
        self.image_size = self.vision_model.config.image_size
        self.grid_size = self.image_size // self.patch_size

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        out = self.vision_model(pixel_values=x)
        patches = out.last_hidden_state  # (B, num_patches, hidden_size)
        feature_map = einops.rearrange(
            patches, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
        )
        return {"feature_map": feature_map}


class Dinov2BackboneWrapper(nn.Module):
    """Wraps a DINOv2 vision model to produce 2D feature maps.

    DINOv2 outputs (B, 1 + num_patches, hidden_size) — a CLS token followed by patch tokens.
    This wrapper strips the CLS token, reshapes patch tokens to (B, hidden_size, H, W),
    and returns {"feature_map": ...} to match the ResNet backbone interface.
    """

    def __init__(self, model_name: str, image_size: int = 224):
        super().__init__()
        if Dinov2Model is None:
            raise ImportError(
                "DINOv2 backbone requires the `transformers` library. "
                "Install it with: pip install transformers"
            )
        self.vision_model = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size
        self.image_size = image_size
        self.grid_size = self.image_size // self.patch_size

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        out = self.vision_model(pixel_values=x)
        patches = out.last_hidden_state[:, 1:]  # strip CLS token → (B, num_patches, hidden_size)
        feature_map = einops.rearrange(
            patches, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
        )
        return {"feature_map": feature_map}


class MoCoV3BackboneWrapper(nn.Module):
    """Wraps a torchvision VisionTransformer loaded with MoCo v3 pretrained weights.

    Produces 2D feature maps by stripping the CLS token and reshaping the patch tokens
    to (B, hidden_dim, H, W), matching the ResNet backbone interface.

    ViT-S/16 (default): hidden_dim=384, 12 layers, 12 heads, mlp_dim=1536, 224px input.
    """

    PRESETS = {
        "vit_small": {"hidden_dim": 384, "num_layers": 12, "num_heads": 12, "mlp_dim": 1536},
        "vit_base": {"hidden_dim": 768, "num_layers": 12, "num_heads": 12, "mlp_dim": 3072},
    }

    def __init__(
        self,
        checkpoint_path: str,
        arch: str = "vit_small",
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        from torchvision.models.vision_transformer import VisionTransformer

        preset = self.PRESETS.get(arch)
        if preset is None:
            raise ValueError(f"Unknown MoCo v3 arch {arch!r}. Choose from {list(self.PRESETS)}")

        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=1000,
            **preset,
        )

        load_mocov3_weights_into_vit(self.vit, checkpoint_path)

        self.hidden_size = preset["hidden_dim"]
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        # Run through ViT internals to get patch features (skip classification head)
        x = self.vit._process_input(x)  # (B, num_patches, hidden_dim)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)  # (B, 1 + num_patches, hidden_dim)

        patches = x[:, 1:]  # strip CLS token
        feature_map = einops.rearrange(
            patches, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
        )
        return {"feature_map": feature_map}


class VoltronBackboneWrapper(nn.Module):
    """Wraps a Voltron (Karamcheti et al. 2023) pretrained encoder.

    Requires the optional ``voltron-robotics`` package. The wrapper calls
    ``voltron.load(model_id)`` to download the checkpoint on first use, then exposes
    the visual-only representation reshaped to a (B, C, H, W) feature map.

    Supported model IDs (all ViT-S/16, 224x224, embed_dim=384 unless noted):
        - "v-cond":       V-Cond ViT-S  (language-conditioned, single-frame)
        - "v-dual":       V-Dual ViT-S  (language-conditioned, dual-frame)
        - "v-gen":        V-Gen  ViT-S
        - "v-cond-base":  V-Cond ViT-B  (embed_dim=768)
        - "r-mvp":        MVP   reproduction (ViT-S, no language)
        - "r-r3m-vit":    R3M   reproduction with ViT-S
    """

    def __init__(
        self,
        model_id: str = "v-cond",
        cache_dir: str | None = None,
    ):
        super().__init__()
        try:
            import voltron
        except ImportError as e:
            raise ImportError(
                "Voltron backbone requires the `voltron-robotics` package. "
                "Install with `pip install voltron-robotics`."
            ) from e

        kwargs = {"freeze": False}
        if cache_dir is not None:
            kwargs["cache"] = cache_dir
        model, _preprocess = voltron.load(model_id, **kwargs)
        self.model = model
        self.model_id = model_id
        self.hidden_size = model.embed_dim
        self.patch_size = getattr(model, "patch_size", 16)
        self.image_size = getattr(model, "resolution", 224)
        self.grid_size = self.image_size // self.patch_size
        self._mode = "visual"

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        representations = self.model.get_representations(x, language=None, mode=self._mode)
        feature_map = einops.rearrange(
            representations, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
        )
        return {"feature_map": feature_map}


class CpMaeBackboneWrapper(nn.Module):
    """Wraps a pretrained CP-MAE ViT encoder.

    Produces feature maps: {"feature_map": (B, embed_dim, H, W)}
    where H=W=14 for ViT-S/16 on 224x224.
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        n_heads: int = 6,
    ):
        super().__init__()
        self.hidden_size = embed_dim
        self.patch_size = patch_size
        self.image_size = img_size
        self.grid_size = img_size // patch_size

        import sys
        from pathlib import Path as _Path

        _this_file = _Path(__file__).resolve()
        # .../src/lerobot/utils/vit_backbones.py -> repo root is 3 levels up from src/lerobot
        _candidates = (
            _this_file.parents[3],  # repo root when layout is <repo>/src/lerobot/utils/...
            _this_file.parents[2],  # fallback if layout differs
        )
        _found = False
        for _candidate in _candidates:
            if (_candidate / "scripts" / "cpmae" / "pretrain_mae.py").exists():
                _repo_root = str(_candidate)
                if _repo_root not in sys.path:
                    sys.path.insert(0, _repo_root)
                _found = True
                break
        if not _found:
            raise FileNotFoundError(
                "Could not locate scripts/cpmae/pretrain_mae.py relative to "
                f"{_this_file}. Searched: {[str(c) for c in _candidates]}"
            )

        from scripts.cpmae.pretrain_mae import ViTEncoder

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
        )

        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(
                f"CP-MAE checkpoint not found at {checkpoint_path}. "
                "Run pretrain_mae.py first to generate the encoder checkpoint."
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        encoded, _ = self.encoder(x, mask=None)
        patches = encoded[:, 1:]  # strip CLS token
        feature_map = einops.rearrange(
            patches, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
        )
        return {"feature_map": feature_map}


def build_vit_backbone(config) -> nn.Module:
    """Build a ViT backbone wrapper from a policy config object.

    Works with any config that has the standard ViT fields
    (vision_backbone, dinov2_model_name, siglip_model_name, etc.).
    """
    if config.vision_backbone.startswith("siglip"):
        return SiglipBackboneWrapper(config.siglip_model_name)
    elif config.vision_backbone.startswith("dinov2"):
        return Dinov2BackboneWrapper(config.dinov2_model_name)
    elif config.vision_backbone.startswith("mocov3"):
        return MoCoV3BackboneWrapper(
            checkpoint_path=config.mocov3_checkpoint_path,
            arch=config.mocov3_arch,
        )
    elif config.vision_backbone.startswith("voltron"):
        return VoltronBackboneWrapper(
            model_id=config.voltron_model_id,
            cache_dir=config.voltron_cache_dir,
        )
    elif config.vision_backbone.startswith("cpmae"):
        return CpMaeBackboneWrapper(
            checkpoint_path=config.cpmae_checkpoint_path,
            img_size=config.cpmae_img_size,
            patch_size=config.cpmae_patch_size,
            embed_dim=config.cpmae_embed_dim,
            depth=config.cpmae_depth,
            n_heads=config.cpmae_n_heads,
        )
    else:
        raise ValueError(f"No ViT wrapper for backbone prefix: {config.vision_backbone}")
