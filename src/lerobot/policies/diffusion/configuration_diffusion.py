#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.backbone_input_norm import VALID_PRESETS as _BACKBONE_NORM_PRESETS
from lerobot.utils.backbone_input_norm import resolve_preset as _resolve_backbone_norm_preset

_logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        freeze_backbone: If True, freeze vision backbone weights during training (no gradient updates).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoders_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    # SSL-pretrained checkpoint (local path or URL) for ResNet backbones.
    # Supports MoCo v1/v2/v3, SimCLR, BYOL, VISSL, and solo-learn formats.
    # Requires use_group_norm=False to preserve BatchNorm weights from SSL pretraining.
    ssl_checkpoint_path: str | None = None
    # SigLIP backbone (used when vision_backbone starts with "siglip").
    siglip_model_name: str | None = None
    # DINOv2 backbone (used when vision_backbone starts with "dinov2").
    dinov2_model_name: str | None = None
    # MoCo v3 ViT backbone (used when vision_backbone starts with "mocov3").
    mocov3_checkpoint_path: str | None = None
    mocov3_arch: str = "vit_small"
    # Voltron backbone (used when vision_backbone starts with "voltron").
    voltron_model_id: str = "v-cond"
    voltron_cache_dir: str | None = None
    # CP-MAE backbone (used when vision_backbone starts with "cpmae").
    cpmae_checkpoint_path: str | None = None
    cpmae_img_size: int = 224
    cpmae_patch_size: int = 16
    cpmae_embed_dim: int = 384
    cpmae_depth: int = 12
    cpmae_n_heads: int = 6
    use_group_norm: bool = True
    # Per-backbone pretraining input normalization applied inside the model, AFTER any
    # dataset-statistic normalization. Use this to match the pixel statistics the backbone
    # was pretrained with (e.g. "imagenet" for supervised/MoCo/SimCLR/BYOL/VIP ResNets).
    # When set to anything other than "identity", `normalization_mapping["VISUAL"]` is
    # automatically forced to IDENTITY to prevent double normalization.
    # Choices: "identity" | "imagenet" | "siglip" | "custom".
    backbone_input_norm: str = "identity"
    backbone_input_mean: tuple[float, float, float] | None = None
    backbone_input_std: tuple[float, float, float] | None = None
    freeze_backbone: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Per-camera heterogeneous backbone: assign a different vision backbone to a
    # specific camera key (e.g. the wrist camera).  When set, the policy
    # automatically enables use_separate_rgb_encoder_per_camera=True and builds
    # the specified camera's encoder with a different backbone than the default.
    # The key must match an entry in input_features (e.g. "observation.images.image2").
    per_camera_backbone: dict[str, str] | None = None
    per_camera_backbone_norm: dict[str, str] | None = None
    per_camera_siglip_model_name: dict[str, str] | None = None
    per_camera_dinov2_model_name: dict[str, str] | None = None
    per_camera_mocov3_checkpoint_path: dict[str, str] | None = None
    per_camera_mocov3_arch: dict[str, str] | None = None
    per_camera_voltron_model_id: dict[str, str] | None = None
    per_camera_voltron_cache_dir: dict[str, str] | None = None
    per_camera_cpmae_checkpoint_path: dict[str, str] | None = None
    per_camera_cpmae_embed_dim: dict[str, int] | None = None
    per_camera_cpmae_n_heads: dict[str, int] | None = None
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Multi-task conditioning.
    num_tasks: int | None = None  # None = single-task (original behavior)
    task_embed_dim: int = 64  # Task embedding dimension
    task_index_offset: int = 0  # Subtracted from task_index before embedding lookup (for per-suite training)

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        supported_prefixes = ("resnet", "siglip", "dinov2", "mocov3", "voltron", "cpmae")
        if not any(self.vision_backbone.startswith(p) for p in supported_prefixes):
            raise ValueError(
                f"`vision_backbone` must start with one of {supported_prefixes}. "
                f"Got {self.vision_backbone}."
            )

        is_resnet = self.vision_backbone.startswith("resnet")

        # Auto-disable ResNet-specific options for ViT backbones.
        if not is_resnet:
            if self.use_group_norm:
                _logger.warning(
                    "use_group_norm=True is ResNet-specific; forcing to False for %s backbone.",
                    self.vision_backbone,
                )
                self.use_group_norm = False
            if self.crop_shape is not None:
                _logger.warning(
                    "crop_shape=%r is designed for ResNet; forcing to None for %s backbone "
                    "(ViT wrappers handle their own resizing).",
                    self.crop_shape, self.vision_backbone,
                )
                self.crop_shape = None
            if self.pretrained_backbone_weights is not None:
                _logger.warning(
                    "pretrained_backbone_weights=%r is ignored for non-ResNet backbones; "
                    "forcing to None for %s backbone.",
                    self.pretrained_backbone_weights, self.vision_backbone,
                )
                self.pretrained_backbone_weights = None

        if self.ssl_checkpoint_path is not None:
            if not is_resnet:
                raise ValueError(
                    "`ssl_checkpoint_path` is only supported for ResNet backbones. "
                    f"Got vision_backbone={self.vision_backbone!r}."
                )
            if self.pretrained_backbone_weights is not None:
                self.pretrained_backbone_weights = None
            if self.use_group_norm:
                self.use_group_norm = False

        # Required-field checks for ViT backbones.
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
        if self.vision_backbone.startswith("cpmae") and self.cpmae_img_size % self.cpmae_patch_size != 0:
            raise ValueError(
                f"`cpmae_img_size` ({self.cpmae_img_size}) must be divisible by "
                f"`cpmae_patch_size` ({self.cpmae_patch_size})."
            )

        # Validate the backbone input-normalization preset and auto-disable the pipeline
        # VISUAL normalization when a pretraining preset is selected, to avoid applying
        # dataset stats on top of the backbone's expected pretraining stats.
        if self.backbone_input_norm not in _BACKBONE_NORM_PRESETS:
            raise ValueError(
                f"`backbone_input_norm` must be one of {_BACKBONE_NORM_PRESETS}. "
                f"Got {self.backbone_input_norm!r}."
            )
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

        # Per-camera heterogeneous backbone: auto-enable separate encoders.
        if self.per_camera_backbone:
            if not self.use_separate_rgb_encoder_per_camera:
                _logger.info(
                    "per_camera_backbone is set; auto-enabling use_separate_rgb_encoder_per_camera=True."
                )
                self.use_separate_rgb_encoder_per_camera = True
            for cam_key, bb in self.per_camera_backbone.items():
                if not any(bb.startswith(p) for p in supported_prefixes):
                    raise ValueError(
                        f"per_camera_backbone['{cam_key}'] must start with one of {supported_prefixes}. "
                        f"Got {bb!r}."
                    )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

        # Validate per-camera backbone keys match actual image features.
        if self.per_camera_backbone:
            valid_cam_keys = set(self.image_features.keys())
            for cam_key in self.per_camera_backbone:
                if cam_key not in valid_cam_keys:
                    raise ValueError(
                        f"per_camera_backbone key '{cam_key}' does not match any image feature. "
                        f"Available camera keys: {sorted(valid_cam_keys)}"
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
