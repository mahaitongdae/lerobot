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


class SD3VaeBackboneWrapper(nn.Module):
    """Wraps an SD3/SDXL/FLUX VAE encoder as a frozen image feature extractor.

    Maps (B, 3, H, W) images to (B, hidden_size, H/8, W/8) feature maps.
    The VAE encoder is always frozen internally; an optional trainable 1x1
    projection expands the thin latent channels (16 for SD3/FLUX, 4 for SDXL)
    to a richer representation for downstream SpatialSoftmax.

    The frozen encoder processes images in sub-batches of ``encode_batch_size``
    to bound peak activation memory (the convolutional encoder keeps full
    spatial resolution at early layers, unlike ViTs that patchify immediately).
    """

    def __init__(
        self,
        model_name: str,
        subfolder: str = "vae",
        latent_proj_dim: int = 256,
        encode_batch_size: int = 8,
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
        self.vae_encoder = vae.encoder
        self.vae_quant_conv = vae.quant_conv  # None in newer diffusers
        self.vae_encoder.requires_grad_(False)
        if self.vae_quant_conv is not None:
            self.vae_quant_conv.requires_grad_(False)

        self._latent_ch = vae.config.latent_channels
        self._scaling = vae.config.scaling_factor
        self._shift = getattr(vae.config, "shift_factor", None) or 0.0
        self._encode_bs = encode_batch_size

        if latent_proj_dim and latent_proj_dim != self._latent_ch:
            self.latent_proj = nn.Conv2d(self._latent_ch, latent_proj_dim, 1)
            self.hidden_size = latent_proj_dim
        else:
            self.latent_proj = None
            self.hidden_size = self._latent_ch

    def _encode_chunk(self, x: Tensor) -> Tensor:
        h = self.vae_encoder(x)
        if self.vae_quant_conv is not None:
            h = self.vae_quant_conv(h)
        return h[:, : self._latent_ch]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = x * 2.0 - 1.0
        B = x.shape[0]
        with torch.no_grad():
            if B <= self._encode_bs:
                z = self._encode_chunk(x)
            else:
                z = torch.cat(
                    [self._encode_chunk(x[i : i + self._encode_bs])
                     for i in range(0, B, self._encode_bs)],
                    dim=0,
                )
        z = (z.detach() - self._shift) * self._scaling
        if self.latent_proj is not None:
            z = self.latent_proj(z)
        return {"feature_map": z}


class WanVaeBackboneWrapper(nn.Module):
    """Wraps a WAN 2.1/2.2 spatiotemporal VAE encoder as a frozen image feature extractor.

    For single-frame input the temporal dimension is unsqueezed before encoding
    and squeezed after. Maps (B, 3, H, W) to (B, hidden_size, H/8, W/8).
    Per-channel latent normalization uses WAN 2.1 default statistics.

    Uses chunked encoding (same as SD3VaeBackboneWrapper) to bound VRAM.
    """

    _DEFAULT_LATENT_MEAN = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    _DEFAULT_LATENT_STD = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]

    def __init__(
        self,
        model_name: str,
        subfolder: str = "vae",
        latent_proj_dim: int = 256,
        encode_batch_size: int = 8,
    ):
        super().__init__()
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(model_name, subfolder=subfolder)
        self.vae_encoder = vae.encoder
        self.vae_quant_conv = vae.quant_conv  # None in newer diffusers
        self.vae_encoder.requires_grad_(False)
        if self.vae_quant_conv is not None:
            self.vae_quant_conv.requires_grad_(False)

        self._latent_ch = getattr(vae.config, "z_dim", None) or vae.config.latent_channels
        self._encode_bs = encode_batch_size

        latent_mean = getattr(vae.config, "latents_mean", None) or self._DEFAULT_LATENT_MEAN
        latent_std = getattr(vae.config, "latents_std", None) or self._DEFAULT_LATENT_STD
        self.register_buffer(
            "_latent_mean", torch.tensor(latent_mean).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "_latent_inv_std",
            (1.0 / torch.tensor(latent_std)).view(1, -1, 1, 1),
        )

        if latent_proj_dim and latent_proj_dim != self._latent_ch:
            self.latent_proj = nn.Conv2d(self._latent_ch, latent_proj_dim, 1)
            self.hidden_size = latent_proj_dim
        else:
            self.latent_proj = None
            self.hidden_size = self._latent_ch

    def _encode_chunk(self, x: Tensor) -> Tensor:
        """Encode a single sub-batch: (b, 3, H, W) → (b, latent_ch, H/8, W/8)."""
        x5d = x.unsqueeze(2)  # (b, 3, 1, H, W)
        h = self.vae_encoder(x5d)
        if self.vae_quant_conv is not None:
            h = self.vae_quant_conv(h)
        return h[:, : self._latent_ch, 0]  # squeeze temporal

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = x * 2.0 - 1.0
        B = x.shape[0]
        with torch.no_grad():
            if B <= self._encode_bs:
                z = self._encode_chunk(x)
            else:
                z = torch.cat(
                    [self._encode_chunk(x[i : i + self._encode_bs])
                     for i in range(0, B, self._encode_bs)],
                    dim=0,
                )
        z = z.detach()
        z = (z - self._latent_mean) * self._latent_inv_std
        if self.latent_proj is not None:
            z = self.latent_proj(z)
        return {"feature_map": z}


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
    elif config.vision_backbone.startswith("sd3vae"):
        return SD3VaeBackboneWrapper(
            model_name=config.sd3vae_model_name,
            subfolder=config.sd3vae_subfolder,
            latent_proj_dim=config.vae_latent_proj_dim,
            encode_batch_size=config.vae_encode_batch_size,
        )
    elif config.vision_backbone.startswith("wanvae"):
        return WanVaeBackboneWrapper(
            model_name=config.wanvae_model_name,
            subfolder=config.wanvae_subfolder,
            latent_proj_dim=config.vae_latent_proj_dim,
            encode_batch_size=config.vae_encode_batch_size,
        )
    else:
        raise ValueError(f"No ViT wrapper for backbone prefix: {config.vision_backbone}")
