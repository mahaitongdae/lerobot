"""Tests for ACT policy with SigLIP vision backbone."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, SiglipBackboneWrapper
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from tests.utils import require_package


def _make_siglip_config(
    image_size=224,
    num_cameras=2,
    state_dim=8,
    action_dim=7,
    chunk_size=10,
    dim_model=64,
    n_heads=4,
    dim_feedforward=128,
    n_encoder_layers=2,
    n_decoder_layers=1,
    n_vae_encoder_layers=2,
    latent_dim=16,
    use_vae=True,
):
    input_features = {}
    for i in range(num_cameras):
        name = f"{OBS_IMAGES}.cam{i}" if num_cameras > 1 else f"{OBS_IMAGES}.image"
        input_features[name] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        vision_backbone="siglip",
        siglip_model_name="google/siglip-base-patch16-224",
        pretrained_backbone_weights=None,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_vae_encoder_layers=n_vae_encoder_layers,
        latent_dim=latent_dim,
        use_vae=use_vae,
        device="cpu",
    )


class FakeSiglipConfig:
    hidden_size = 64
    patch_size = 16
    image_size = 224


class FakeSiglipOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeSiglipVisionModel(nn.Module):
    """Lightweight stand-in for SiglipVisionModel to avoid downloading weights."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = FakeSiglipConfig()
        grid = self.config.image_size // self.config.patch_size  # 14
        self.num_patches = grid * grid
        self.proj = nn.Linear(3 * self.config.patch_size * self.config.patch_size, self.config.hidden_size)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, pixel_values, **kwargs):
        b = pixel_values.shape[0]
        patches = pixel_values.unfold(2, self.config.patch_size, self.config.patch_size).unfold(
            3, self.config.patch_size, self.config.patch_size
        )
        patches = patches.contiguous().view(b, 3, self.num_patches, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.num_patches, -1)
        hidden = self.proj(patches)
        return FakeSiglipOutput(last_hidden_state=hidden)


@pytest.fixture
def fake_siglip():
    """Patch SiglipVisionModel.from_pretrained to use FakeSiglipVisionModel."""
    with patch.object(SiglipBackboneWrapper, "__init__", _fake_siglip_init):
        yield


def _fake_siglip_init(self, model_name: str):
    nn.Module.__init__(self)
    fake = FakeSiglipVisionModel()
    self.vision_model = fake
    self.hidden_size = fake.config.hidden_size
    self.patch_size = fake.config.patch_size
    self.image_size = fake.config.image_size
    self.grid_size = self.image_size // self.patch_size


def _make_batch(config, batch_size=2):
    batch = {}
    for key, feat in config.input_features.items():
        if feat.type is FeatureType.VISUAL:
            batch[key] = torch.randn(batch_size, *feat.shape)
        elif feat.type is FeatureType.STATE:
            batch[key] = torch.randn(batch_size, *feat.shape)
    batch[ACTION] = torch.randn(batch_size, config.chunk_size, config.action_feature.shape[0])
    batch["action_is_pad"] = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
    batch[OBS_IMAGES] = [batch[key] for key in config.image_features]
    return batch


# ── Config validation ────────────────────────────────────────────────


def test_siglip_config_requires_model_name():
    with pytest.raises(ValueError, match="siglip_model_name"):
        ACTConfig(
            vision_backbone="siglip",
            siglip_model_name=None,
            input_features={
                f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


def test_siglip_config_valid():
    config = _make_siglip_config()
    assert config.vision_backbone == "siglip"
    assert config.siglip_model_name == "google/siglip-base-patch16-224"


def test_resnet_config_still_works():
    config = ACTConfig(
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        input_features={
            f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    assert config.vision_backbone == "resnet18"


def test_unsupported_backbone_rejected():
    with pytest.raises(ValueError, match="must start with one of"):
        ACTConfig(
            vision_backbone="vit_base",
            input_features={
                f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


# ── SiglipBackboneWrapper ───────────────────────────────────────────


def test_siglip_wrapper_output_shape(fake_siglip):
    wrapper = SiglipBackboneWrapper("google/siglip-base-patch16-224")
    x = torch.randn(2, 3, 224, 224)
    out = wrapper(x)
    assert "feature_map" in out
    fm = out["feature_map"]
    assert fm.shape == (2, FakeSiglipConfig.hidden_size, 14, 14)


def test_siglip_wrapper_auto_resizes(fake_siglip):
    wrapper = SiglipBackboneWrapper("google/siglip-base-patch16-224")
    x = torch.randn(1, 3, 256, 256)
    out = wrapper(x)
    assert out["feature_map"].shape == (1, FakeSiglipConfig.hidden_size, 14, 14)


# ── ACT model with SigLIP ───────────────────────────────────────────


def test_act_siglip_instantiation(fake_siglip):
    config = _make_siglip_config()
    model = ACT(config)
    assert isinstance(model.backbone.backbone, SiglipBackboneWrapper)
    assert hasattr(model, "encoder_img_feat_input_proj")
    assert hasattr(model, "encoder_cam_feat_pos_embed")


def test_act_siglip_forward(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    model = ACT(config)
    model.train()
    batch = _make_batch(config, batch_size=2)
    actions, (mu, log_sigma_x2) = model(batch)
    assert actions.shape == (2, 5, 7)
    assert mu.shape == (2, config.latent_dim)
    assert log_sigma_x2.shape == (2, config.latent_dim)


def test_act_siglip_forward_no_vae(fake_siglip):
    config = _make_siglip_config(chunk_size=5, use_vae=False)
    model = ACT(config)
    model.eval()
    batch = _make_batch(config, batch_size=2)
    del batch[ACTION]
    del batch["action_is_pad"]
    actions, (mu, log_sigma_x2) = model(batch)
    assert actions.shape == (2, 5, 7)
    assert mu is None
    assert log_sigma_x2 is None


def test_act_siglip_no_nan_in_output(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    model = ACT(config)
    model.train()
    batch = _make_batch(config, batch_size=2)
    actions, (mu, log_sigma_x2) = model(batch)
    assert not torch.isnan(actions).any()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_sigma_x2).any()


# ── ACTPolicy with SigLIP ───────────────────────────────────────────


def test_policy_forward_and_loss(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.train()
    batch = _make_batch(config, batch_size=2)
    del batch[OBS_IMAGES]
    loss, loss_dict = policy.forward(batch)
    assert loss.dim() == 0
    assert "l1_loss" in loss_dict
    assert "kld_loss" in loss_dict


def test_policy_select_action(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.eval()

    obs = {}
    for key, feat in config.input_features.items():
        obs[key] = torch.randn(1, *feat.shape)

    action = policy.select_action(obs)
    assert action.shape == (1, 7)


def test_policy_backward(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.train()
    batch = _make_batch(config, batch_size=2)
    del batch[OBS_IMAGES]
    loss, _ = policy.forward(batch)
    loss.backward()
    backbone_params = [p for n, p in policy.named_parameters() if "backbone" in n]
    assert len(backbone_params) > 0
    assert all(p.grad is not None for p in backbone_params if p.requires_grad)


def test_policy_optim_param_groups(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    config.optimizer_lr_backbone = 1e-6
    policy = ACTPolicy(config)
    groups = policy.get_optim_params()
    assert len(groups) == 2
    backbone_group = groups[1]
    assert backbone_group["lr"] == 1e-6
    assert len(backbone_group["params"]) > 0


# ── Freeze backbone ──────────────────────────────────────────────────


def test_freeze_backbone_disables_grads(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    config.freeze_backbone = True
    policy = ACTPolicy(config)
    for p in policy.model.backbone.parameters():
        assert not p.requires_grad


def test_freeze_backbone_excludes_from_optim(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    config.freeze_backbone = True
    policy = ACTPolicy(config)
    groups = policy.get_optim_params()
    backbone_group = groups[1]
    assert len(backbone_group["params"]) == 0


def test_freeze_backbone_no_grad_after_backward(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    config.freeze_backbone = True
    policy = ACTPolicy(config)
    policy.train()
    batch = _make_batch(config, batch_size=2)
    del batch[OBS_IMAGES]
    loss, _ = policy.forward(batch)
    loss.backward()
    for p in policy.model.backbone.parameters():
        assert p.grad is None


def test_unfreeze_backbone_keeps_grads(fake_siglip):
    config = _make_siglip_config(chunk_size=5)
    config.freeze_backbone = False
    policy = ACTPolicy(config)
    for p in policy.model.backbone.parameters():
        assert p.requires_grad


# ── Single camera ───────────────────────────────────────────────────


def test_act_siglip_single_camera(fake_siglip):
    config = _make_siglip_config(num_cameras=1, chunk_size=5)
    model = ACT(config)
    model.train()
    batch = _make_batch(config, batch_size=2)
    actions, _ = model(batch)
    assert actions.shape == (2, 5, 7)


# ── Save and load ───────────────────────────────────────────────────


def test_save_and_load_pretrained(fake_siglip, tmp_path):
    config = _make_siglip_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.save_pretrained(tmp_path / "act_siglip")
    loaded = ACTPolicy.from_pretrained(tmp_path / "act_siglip", config=config)
    for p_orig, p_loaded in zip(policy.parameters(), loaded.parameters()):
        torch.testing.assert_close(p_orig, p_loaded, rtol=0, atol=0)
