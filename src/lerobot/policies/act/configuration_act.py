#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.backbone_input_norm import VALID_PRESETS as _BACKBONE_NORM_PRESETS
from lerobot.utils.backbone_input_norm import resolve_preset as _resolve_backbone_norm_preset

_logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the vision backbone to use for encoding images. Supports torchvision
            ResNet variants (e.g. "resnet18"), "siglip" (requires `siglip_model_name`), "dinov2"
            (requires `dinov2_model_name`), "mocov3" (requires `mocov3_checkpoint_path`),
            "voltron" (requires `voltron_model_id` and the optional `voltron-robotics` package),
            or "cpmae" (requires `cpmae_checkpoint_path`).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights. Only used for ResNet backbones.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution. Only used for ResNet backbones.
        siglip_model_name: HuggingFace model name for SigLIP (e.g. "google/siglip-base-patch16-224").
            Required when `vision_backbone` starts with "siglip".
        dinov2_model_name: HuggingFace model name for DINOv2 (e.g. "facebook/dinov2-small").
            Required when `vision_backbone` starts with "dinov2".
        mocov3_checkpoint_path: Local path or URL to a MoCo v3 ViT checkpoint (.pth.tar).
            Required when `vision_backbone` starts with "mocov3".
        mocov3_arch: MoCo v3 ViT architecture variant ("vit_small" or "vit_base").
        voltron_model_id: Voltron model id (e.g. "v-cond" for ViT-S, "v-cond-base" for ViT-B).
            Required when `vision_backbone` starts with "voltron".
        voltron_cache_dir: Local directory for the `voltron-robotics` package to cache
            downloaded configs/checkpoints. Defaults to the package-internal "cache/" folder.
        freeze_backbone: If True, freeze vision backbone weights during training (no gradient updates).
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # SSL-pretrained checkpoint (local path or URL) for ResNet backbones.
    # Supports MoCo v1/v2/v3, SimCLR, BYOL, VISSL, and solo-learn formats.
    # Key prefixes are auto-detected and stripped. Mutually exclusive with pretrained_backbone_weights.
    ssl_checkpoint_path: str | None = None
    # SigLIP backbone (used when vision_backbone starts with "siglip").
    siglip_model_name: str | None = None
    # DINOv2 backbone (used when vision_backbone starts with "dinov2").
    dinov2_model_name: str | None = None
    # MoCo v3 ViT backbone (used when vision_backbone starts with "mocov3").
    mocov3_checkpoint_path: str | None = None
    mocov3_arch: str = "vit_small"
    # Voltron backbone (used when vision_backbone starts with "voltron").
    # Requires the optional `voltron-robotics` package.
    voltron_model_id: str = "v-cond"
    voltron_cache_dir: str | None = None
    # CP-MAE backbone (used when vision_backbone starts with "cpmae").
    cpmae_checkpoint_path: str | None = None
    cpmae_img_size: int = 224
    cpmae_patch_size: int = 16
    cpmae_embed_dim: int = 384
    cpmae_depth: int = 12
    cpmae_n_heads: int = 6
    # V-JEPA 2 / 2.1 backbone (used when vision_backbone starts with "vjepa2").
    vjepa2_repo_or_dir: str = "facebookresearch/vjepa2"
    vjepa2_model_name: str = "vjepa2_1_vit_base_384"
    # Optional local path or URL. If None, uses Meta's public fbaipublicfiles checkpoint URL.
    vjepa2_checkpoint_url: str | None = None
    vjepa2_input_frames: int = 1
    vjepa2_spatial_pool_size: int | None = None
    # SD3/SDXL/FLUX VAE backbone (used when vision_backbone starts with "sd3vae").
    sd3vae_model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    sd3vae_subfolder: str = "vae"
    # Feature tensor to expose from the SD3 VAE encoder. "latent_mean" preserves
    # the original behavior; values like "latent_moments", "mid_block", and
    # "down_blocks.2" enable control-oriented feature probes.
    sd3vae_feature_layer: str = "latent_mean"
    # Trainable 1x1 projection dim for generation VAE latents (sd3vae/wanvae).
    vae_latent_proj_dim: int | None = 256
    vae_encode_batch_size: int = 64
    # Spatial pooling for VAE latents. The SD3 VAE produces 28×28 spatial maps from
    # 224px input; ACT flattens these to sequence tokens, causing O(n²) attention OOM.
    # Set to 14 to match ViT-style token counts (14×14=196 tokens per camera).
    vae_spatial_pool_size: int | None = 14
    # Per-backbone pretraining input normalization applied inside the model, AFTER any
    # dataset-statistic normalization. Use this to match the pixel statistics the backbone
    # was pretrained with (e.g. "imagenet" for ResNet/DINOv2/MoCov3/MVP/VC-1/Voltron,
    # "siglip" for SigLIP, "identity" for backbones pretrained on raw [0,1] such as CP-MAE).
    # When set to anything other than "identity", `normalization_mapping["VISUAL"]` is
    # automatically forced to IDENTITY to prevent double normalization.
    # Choices: "identity" | "imagenet" | "siglip" | "custom".
    backbone_input_norm: str = "identity"
    backbone_input_mean: tuple[float, float, float] | None = None
    backbone_input_std: tuple[float, float, float] | None = None
    freeze_backbone: bool = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Multi-task conditioning.
    num_tasks: int | None = None  # None = single-task (original behavior)
    task_embed_dim: int = 64  # Task embedding dimension
    task_index_offset: int = 0  # Subtracted from task_index before embedding lookup (for per-suite training)
    use_task_film_on_vision: bool = False  # Ablation: FiLM on vision features

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        supported_prefixes = (
            "resnet",
            "siglip",
            "dinov2",
            "mocov3",
            "voltron",
            "cpmae",
            "vjepa2",
            "sd3vae",
        )
        if not any(self.vision_backbone.startswith(p) for p in supported_prefixes):
            raise ValueError(
                f"`vision_backbone` must start with one of {supported_prefixes}. Got {self.vision_backbone}."
            )
        if self.ssl_checkpoint_path is not None:
            if not self.vision_backbone.startswith("resnet"):
                raise ValueError(
                    "`ssl_checkpoint_path` is only supported for ResNet backbones. "
                    f"Got vision_backbone={self.vision_backbone!r}."
                )
            if self.pretrained_backbone_weights is not None:
                self.pretrained_backbone_weights = None
        if self.vision_backbone.startswith("siglip") and not self.siglip_model_name:
            raise ValueError(
                "`siglip_model_name` must be set when using a SigLIP vision backbone "
                "(e.g. 'google/siglip-base-patch16-224')."
            )
        if self.vision_backbone.startswith("dinov2") and not self.dinov2_model_name:
            raise ValueError(
                "`dinov2_model_name` must be set when using a DINOv2 vision backbone "
                "(e.g. 'facebook/dinov2-small')."
            )
        if self.vision_backbone.startswith("mocov3") and not self.mocov3_checkpoint_path:
            raise ValueError(
                "`mocov3_checkpoint_path` must be set when using a MoCo v3 vision backbone "
                "(e.g. 'https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar')."
            )
        if self.vision_backbone.startswith("voltron") and not self.voltron_model_id:
            raise ValueError(
                "`voltron_model_id` must be set when using a Voltron vision backbone "
                "(e.g. 'v-cond' or 'v-cond-base')."
            )
        if self.vision_backbone.startswith("cpmae") and not self.cpmae_checkpoint_path:
            raise ValueError(
                "`cpmae_checkpoint_path` must be set when using a CP-MAE vision backbone "
                "(e.g. 'results/M3_cpmae/R200_cpmae/encoder_final.pt')."
            )
        if self.vision_backbone.startswith("vjepa2") and not self.vjepa2_repo_or_dir:
            raise ValueError("`vjepa2_repo_or_dir` must be set when using a V-JEPA2 vision backbone.")
        if self.vision_backbone.startswith("vjepa2") and not self.vjepa2_model_name:
            raise ValueError("`vjepa2_model_name` must be set when using a V-JEPA2 vision backbone.")
        if self.vision_backbone.startswith("vjepa2") and self.vjepa2_input_frames < 1:
            raise ValueError(f"`vjepa2_input_frames` must be >= 1. Got {self.vjepa2_input_frames}.")
        if (
            self.vision_backbone.startswith("vjepa2")
            and self.vjepa2_spatial_pool_size is not None
            and self.vjepa2_spatial_pool_size < 1
        ):
            raise ValueError(
                f"`vjepa2_spatial_pool_size` must be >= 1 or None. Got {self.vjepa2_spatial_pool_size}."
            )
        if self.vision_backbone.startswith("sd3vae") and not self.sd3vae_model_name:
            raise ValueError(
                "`sd3vae_model_name` must be set when using an SD3 VAE vision backbone "
                "(e.g. 'stabilityai/stable-diffusion-3-medium-diffusers')."
            )
        if self.vision_backbone.startswith("sd3vae") and not self.sd3vae_feature_layer:
            raise ValueError("`sd3vae_feature_layer` must be non-empty when using an SD3 VAE backbone.")
        if self.vision_backbone.startswith("sd3vae"):
            if self.freeze_backbone:
                _logger.warning(
                    "SD3 VAE encoder is frozen internally. Setting freeze_backbone=True "
                    "also freezes the trainable latent projection; consider freeze_backbone=False."
                )
            if self.backbone_input_norm != "identity":
                _logger.warning(
                    "SD3 VAE handles its own [0,1]->[-1,1] scaling; "
                    "forcing backbone_input_norm from %r to 'identity'.",
                    self.backbone_input_norm,
                )
                self.backbone_input_norm = "identity"

        # Validate the backbone input-normalization preset and auto-disable the pipeline
        # VISUAL normalization when a pretraining preset is selected, to avoid applying
        # dataset stats on top of the backbone's expected pretraining stats.
        if self.backbone_input_norm not in _BACKBONE_NORM_PRESETS:
            raise ValueError(
                f"`backbone_input_norm` must be one of {_BACKBONE_NORM_PRESETS}. "
                f"Got {self.backbone_input_norm!r}."
            )
        # Eagerly validate custom mean/std so we fail fast on misconfiguration.
        _resolve_backbone_norm_preset(
            self.backbone_input_norm, self.backbone_input_mean, self.backbone_input_std
        )
        if self.backbone_input_norm != "identity":
            current = self.normalization_mapping.get("VISUAL", NormalizationMode.IDENTITY)
            if current != NormalizationMode.IDENTITY:
                _logger.warning(
                    "backbone_input_norm=%r is set; forcing normalization_mapping['VISUAL'] "
                    "from %s to IDENTITY to avoid double-normalizing images.",
                    self.backbone_input_norm,
                    current,
                )
                self.normalization_mapping["VISUAL"] = NormalizationMode.IDENTITY

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps < 1:
            raise ValueError(f"`n_obs_steps` must be >= 1. Got {self.n_obs_steps}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> list | None:
        if self.n_obs_steps == 1:
            return None
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        if self.n_obs_steps == 1:
            return list(range(self.chunk_size))
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
