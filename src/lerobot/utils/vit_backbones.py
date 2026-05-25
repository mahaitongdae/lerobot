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


_VJEPA2_PUBLIC_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
_VJEPA2_CHECKPOINT_FILES = {
    "vit_large": "vitl",
    "vit_huge": "vith",
    "vit_giant": "vitg",
    "vit_ac_giant": "vjepa2-ac-vitg",
    "vit_giant_384": "vitg-384",
    "vjepa2_1_vit_base_384": "vjepa2_1_vitb_dist_vitG_384",
    "vjepa2_1_vit_large_384": "vjepa2_1_vitl_dist_vitG_384",
    "vjepa2_1_vit_giant_384": "vjepa2_1_vitg_384",
    "vjepa2_1_vit_gigantic_384": "vjepa2_1_vitG_384",
}


def _clean_vjepa2_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): val
        for key, val in state_dict.items()
    }


def _default_vjepa2_checkpoint_url(model_name: str) -> str:
    try:
        checkpoint_file = _VJEPA2_CHECKPOINT_FILES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No default V-JEPA2 checkpoint URL is known for model_name={model_name!r}. "
            "Set `vjepa2_checkpoint_url` to a local checkpoint path or URL."
        ) from exc
    return f"{_VJEPA2_PUBLIC_BASE_URL}/{checkpoint_file}.pt"


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


class VJepa2BackboneWrapper(nn.Module):
    """Wraps a V-JEPA 2 / 2.1 PyTorch Hub encoder to produce 2D feature maps.

    Meta's PyTorch Hub V-JEPA encoders expect videos as ``(B, C, T, H, W)``.
    For image-only policy observations, this wrapper inserts/repeats the temporal
    dimension, then averages temporal patch groups back to one spatial feature map.
    """

    def __init__(
        self,
        repo_or_dir: str = "facebookresearch/vjepa2",
        model_name: str = "vjepa2_1_vit_base_384",
        input_frames: int = 1,
        spatial_pool_size: int | None = None,
        checkpoint_url: str | None = None,
    ):
        super().__init__()
        if input_frames < 1:
            raise ValueError(f"`input_frames` must be >= 1. Got {input_frames}.")

        from pathlib import Path

        hub_source = "local" if Path(repo_or_dir).exists() else "github"
        # Upstream V-JEPA2 hub currently points pretrained=True at localhost in
        # some snapshots, so instantiate the architecture first and load weights here.
        hub_kwargs = {"source": hub_source, "pretrained": False}
        if hub_source == "github":
            hub_kwargs["trust_repo"] = True

        loaded = torch.hub.load(repo_or_dir, model_name, **hub_kwargs)
        self.encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        self.predictor = loaded[1] if isinstance(loaded, (tuple, list)) and len(loaded) > 1 else None

        checkpoint_url = checkpoint_url or _default_vjepa2_checkpoint_url(model_name)
        checkpoint_path = Path(checkpoint_url)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        else:
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

        encoder_key = "ema_encoder" if model_name.startswith("vjepa2_1") else "target_encoder"
        if encoder_key not in state_dict:
            encoder_key = "encoder"
        encoder_state_dict = _clean_vjepa2_state_dict(state_dict[encoder_key])
        strict = model_name.startswith("vjepa2_1")
        self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        if self.predictor is not None and "predictor" in state_dict:
            predictor_state_dict = _clean_vjepa2_state_dict(state_dict["predictor"])
            self.predictor.load_state_dict(predictor_state_dict, strict=strict)

        self.repo_or_dir = repo_or_dir
        self.model_name = model_name
        self.checkpoint_url = checkpoint_url
        self.input_frames = input_frames

        self.hidden_size = int(
            getattr(self.encoder, "embed_dim", getattr(self.encoder, "num_features", 768))
        )
        self.patch_size = int(getattr(self.encoder, "patch_size", 16))
        self.image_size = int(
            getattr(self.encoder, "img_height", getattr(self.encoder, "image_size", 384))
        )
        self.grid_size = self.image_size // self.patch_size
        self._spatial_tokens = self.grid_size * self.grid_size
        self._pool = nn.AdaptiveAvgPool2d(spatial_pool_size) if spatial_pool_size is not None else None

    @staticmethod
    def _extract_tokens(output) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, dict):
            for key in ("last_hidden_state", "hidden_state", "x"):
                if key in output:
                    return output[key]
        if isinstance(output, (tuple, list)) and output:
            for item in reversed(output):
                if isinstance(item, Tensor):
                    return item
        raise TypeError(f"Could not extract V-JEPA2 token tensor from output type {type(output)!r}.")

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        video = x.unsqueeze(2)
        if self.input_frames > 1:
            video = video.repeat(1, 1, self.input_frames, 1, 1)

        patches = self._extract_tokens(self.encoder(video))
        if patches.ndim != 3:
            raise ValueError(f"Expected V-JEPA2 tokens with shape (B, N, C), got {tuple(patches.shape)}.")

        n_tokens = patches.shape[1]
        if n_tokens == self._spatial_tokens:
            feature_map = einops.rearrange(
                patches, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
            )
        elif n_tokens % self._spatial_tokens == 0:
            n_temporal = n_tokens // self._spatial_tokens
            feature_map = einops.rearrange(
                patches,
                "b (t h w) c -> b c t h w",
                t=n_temporal,
                h=self.grid_size,
                w=self.grid_size,
            ).mean(dim=2)
        else:
            raise ValueError(
                "Cannot reshape V-JEPA2 tokens into a square feature map: "
                f"got {n_tokens} tokens, expected {self._spatial_tokens} or a temporal multiple."
            )

        if self._pool is not None:
            feature_map = self._pool(feature_map)
        return {"feature_map": feature_map}


class SD3VaeBackboneWrapper(nn.Module):
    """Wraps an SD3/SDXL/FLUX VAE encoder as a frozen image feature extractor.

    Maps (B, 3, H, W) images to (B, hidden_size, pool_size, pool_size) feature maps.
    The VAE encoder is always frozen internally; an optional trainable 1x1
    projection expands the selected VAE feature tensor to a richer
    representation for downstream SpatialSoftmax.

    The frozen encoder processes images in sub-batches of ``encode_batch_size``
    to bound peak activation memory (the convolutional encoder keeps full
    spatial resolution at early layers, unlike ViTs that patchify immediately).

    ``feature_layer`` selects which representation to expose:
        - "latent_mean": current behavior, the scaled first latent_channels
          channels after quant_conv.
        - "latent_moments": all post-quant_conv channels, typically mean and
          logvar. The mean half is scaled like "latent_mean".
        - "encoder_out": encoder output before quant_conv.
        - "mid_block" or module paths like "down_blocks.2": intermediate
          encoder activations captured by a forward hook.

    When ``spatial_pool_size`` is set, an adaptive average pool reduces the
    native 28x28 latent (from 224px input) to a smaller grid. This is critical
    for ACT-style policies that flatten spatial tokens into the transformer
    sequence - 28x28=784 tokens causes O(n^2) attention OOM, while 14x14=196
    matches ViT token counts.
    """

    _LATENT_MEAN = "latent_mean"
    _LATENT_MOMENTS = "latent_moments"
    _ENCODER_OUT = "encoder_out"
    _ALIASES = {
        "latent": _LATENT_MEAN,
        "moments": _LATENT_MOMENTS,
        "full_latent": _LATENT_MOMENTS,
        "encoder": _ENCODER_OUT,
        "down1": "down_blocks.1",
        "down2": "down_blocks.2",
        "down3": "down_blocks.3",
        "down_block1": "down_blocks.1",
        "down_block2": "down_blocks.2",
        "down_block3": "down_blocks.3",
        "down_blocks_1": "down_blocks.1",
        "down_blocks_2": "down_blocks.2",
        "down_blocks_3": "down_blocks.3",
    }

    def __init__(
        self,
        model_name: str,
        subfolder: str = "vae",
        latent_proj_dim: int = 256,
        encode_batch_size: int = 8,
        spatial_pool_size: int | None = None,
        feature_layer: str = "latent_mean",
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
        self.vae_encoder = vae.encoder
        self.vae_quant_conv = vae.quant_conv  # None in newer diffusers
        self.feature_layer = self._normalize_feature_layer(feature_layer)
        self.vae_encoder.requires_grad_(False)
        if self.vae_quant_conv is not None:
            self.vae_quant_conv.requires_grad_(False)

        self._latent_ch = vae.config.latent_channels
        self._scaling = vae.config.scaling_factor
        self._shift = getattr(vae.config, "shift_factor", None) or 0.0
        self._encode_bs = encode_batch_size

        if spatial_pool_size is not None:
            self._pool = nn.AdaptiveAvgPool2d(spatial_pool_size)
        else:
            self._pool = None

        feature_ch = self._infer_feature_channels()
        if latent_proj_dim and latent_proj_dim != feature_ch:
            if feature_ch is None:
                self.latent_proj = nn.LazyConv2d(latent_proj_dim, 1)
            else:
                self.latent_proj = nn.Conv2d(feature_ch, latent_proj_dim, 1)
            self.hidden_size = latent_proj_dim
        else:
            if feature_ch is None:
                raise ValueError(
                    "Cannot infer channels for SD3 VAE feature_layer="
                    f"{self.feature_layer!r} with vae_latent_proj_dim=None. "
                    "Set `vae_latent_proj_dim` so a LazyConv2d projection can be used."
                )
            self.latent_proj = None
            self.hidden_size = feature_ch

    @classmethod
    def _normalize_feature_layer(cls, feature_layer: str) -> str:
        normalized = feature_layer.strip()
        return cls._ALIASES.get(normalized, normalized)

    @staticmethod
    def _get_module(root: nn.Module, path: str) -> nn.Module:
        module: nn.Module | nn.ModuleList = root
        for part in path.split("."):
            if part.isdigit():
                try:
                    module = module[int(part)]  # type: ignore[index]
                except (IndexError, TypeError) as exc:
                    raise ValueError(f"Could not resolve SD3 VAE encoder module path {path!r}.") from exc
            else:
                if not hasattr(module, part):
                    raise ValueError(f"Could not resolve SD3 VAE encoder module path {path!r}.")
                module = getattr(module, part)
        if not isinstance(module, nn.Module):
            raise ValueError(f"SD3 VAE encoder path {path!r} did not resolve to a torch module.")
        return module

    @staticmethod
    def _infer_module_out_channels(module: nn.Module | None) -> int | None:
        if module is None:
            return None

        out_channels = getattr(module, "out_channels", None)
        if isinstance(out_channels, int):
            return out_channels

        for attr in ("conv_out", "conv2", "conv", "proj_out"):
            child = getattr(module, attr, None)
            out_channels = getattr(child, "out_channels", None)
            if isinstance(out_channels, int):
                return out_channels

        resnets = getattr(module, "resnets", None)
        if resnets:
            return SD3VaeBackboneWrapper._infer_module_out_channels(resnets[-1])

        return None

    def _infer_feature_channels(self) -> int | None:
        if self.feature_layer == self._LATENT_MEAN:
            return self._latent_ch
        if self.feature_layer == self._LATENT_MOMENTS:
            return self._infer_module_out_channels(self.vae_quant_conv)
        if self.feature_layer == self._ENCODER_OUT:
            return self._infer_module_out_channels(self.vae_encoder)
        return self._infer_module_out_channels(self._get_module(self.vae_encoder, self.feature_layer))

    @staticmethod
    def _first_tensor(output) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        raise TypeError(f"Expected SD3 VAE feature hook output to be a tensor, got {type(output)!r}.")

    def _scale_latent_moments(self, h: Tensor) -> Tensor:
        mean = (h[:, : self._latent_ch] - self._shift) * self._scaling
        if h.shape[1] <= self._latent_ch:
            return mean
        return torch.cat([mean, h[:, self._latent_ch :]], dim=1)

    def _encode_chunk(self, x: Tensor) -> Tensor:
        if self.feature_layer in (self._LATENT_MEAN, self._LATENT_MOMENTS, self._ENCODER_OUT):
            h = self.vae_encoder(x)
            if self.feature_layer == self._ENCODER_OUT:
                return h
            if self.vae_quant_conv is not None:
                h = self.vae_quant_conv(h)
            if self.feature_layer == self._LATENT_MOMENTS:
                return self._scale_latent_moments(h)
            return self._scale_latent_moments(h)[:, : self._latent_ch]

        features: dict[str, Tensor] = {}

        def save_feature(_module, _inputs, output):
            features["value"] = self._first_tensor(output)

        handle = self._get_module(self.vae_encoder, self.feature_layer).register_forward_hook(save_feature)
        try:
            self.vae_encoder(x)
        finally:
            handle.remove()
        if "value" not in features:
            raise RuntimeError(f"SD3 VAE feature hook {self.feature_layer!r} did not capture an output.")
        return features["value"]

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
        if self._pool is not None:
            z = self._pool(z)
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
    elif config.vision_backbone.startswith("vjepa2"):
        return VJepa2BackboneWrapper(
            repo_or_dir=config.vjepa2_repo_or_dir,
            model_name=config.vjepa2_model_name,
            input_frames=config.vjepa2_input_frames,
            spatial_pool_size=getattr(config, "vjepa2_spatial_pool_size", None),
            checkpoint_url=getattr(config, "vjepa2_checkpoint_url", None),
        )
    elif config.vision_backbone.startswith("sd3vae"):
        return SD3VaeBackboneWrapper(
            model_name=config.sd3vae_model_name,
            subfolder=config.sd3vae_subfolder,
            latent_proj_dim=config.vae_latent_proj_dim,
            encode_batch_size=config.vae_encode_batch_size,
            spatial_pool_size=getattr(config, "vae_spatial_pool_size", None),
            feature_layer=getattr(config, "sd3vae_feature_layer", "latent_mean"),
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
