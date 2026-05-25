"""Tests for Diffusion Policy with ViT vision backbones."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
    _FeatureMapDictToTensor,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.vit_backbones import Dinov2BackboneWrapper, SiglipBackboneWrapper, VJepa2BackboneWrapper


# ── Fake backbone stubs ─────────────────────────────────────────────

HIDDEN_SIZE = 64
PATCH_SIZE = 14
IMAGE_SIZE = 224
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 16


class FakeDinov2Config:
    hidden_size = HIDDEN_SIZE
    patch_size = PATCH_SIZE
    image_size = IMAGE_SIZE


class FakeDinov2Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeDinov2Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = FakeDinov2Config()
        self.num_patches = GRID_SIZE * GRID_SIZE
        self.proj = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, HIDDEN_SIZE)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, pixel_values, **kwargs):
        b = pixel_values.shape[0]
        patches = pixel_values.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.contiguous().view(b, 3, self.num_patches, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.num_patches, -1)
        hidden = self.proj(patches)
        cls_token = torch.zeros(b, 1, HIDDEN_SIZE, device=hidden.device)
        hidden = torch.cat([cls_token, hidden], dim=1)
        return FakeDinov2Output(last_hidden_state=hidden)


SIGLIP_HIDDEN = 64
SIGLIP_PATCH = 16
SIGLIP_IMAGE = 224
SIGLIP_GRID = SIGLIP_IMAGE // SIGLIP_PATCH  # 14


class FakeSiglipConfig:
    hidden_size = SIGLIP_HIDDEN
    patch_size = SIGLIP_PATCH
    image_size = SIGLIP_IMAGE


class FakeSiglipOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeSiglipModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = FakeSiglipConfig()
        self.num_patches = SIGLIP_GRID * SIGLIP_GRID
        self.proj = nn.Linear(3 * SIGLIP_PATCH * SIGLIP_PATCH, SIGLIP_HIDDEN)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, pixel_values, **kwargs):
        b = pixel_values.shape[0]
        patches = pixel_values.unfold(2, SIGLIP_PATCH, SIGLIP_PATCH).unfold(3, SIGLIP_PATCH, SIGLIP_PATCH)
        patches = patches.contiguous().view(b, 3, self.num_patches, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.num_patches, -1)
        hidden = self.proj(patches)
        return FakeSiglipOutput(last_hidden_state=hidden)


VJEPA_HIDDEN = 32
VJEPA_PATCH = 16
VJEPA_IMAGE = 384
VJEPA_GRID = VJEPA_IMAGE // VJEPA_PATCH  # 24


class FakeVJepa2Encoder(nn.Module):
    embed_dim = VJEPA_HIDDEN
    patch_size = VJEPA_PATCH
    img_height = VJEPA_IMAGE
    img_width = VJEPA_IMAGE

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, VJEPA_HIDDEN)
        self.last_input_shape = None

    def forward(self, x):
        self.last_input_shape = tuple(x.shape)
        b, _c, t, _h, _w = x.shape
        patches = x.unfold(3, VJEPA_PATCH, VJEPA_PATCH).unfold(4, VJEPA_PATCH, VJEPA_PATCH)
        patch_means = patches.contiguous().mean(dim=(1, 5, 6))
        tokens = patch_means.reshape(b, t * VJEPA_GRID * VJEPA_GRID, 1)
        return self.proj(tokens)


def _fake_dinov2_init(self, model_name: str, image_size: int = 224):
    nn.Module.__init__(self)
    fake = FakeDinov2Model()
    self.vision_model = fake
    self.hidden_size = fake.config.hidden_size
    self.patch_size = fake.config.patch_size
    self.image_size = fake.config.image_size
    self.grid_size = self.image_size // self.patch_size


def _fake_siglip_init(self, model_name: str):
    nn.Module.__init__(self)
    fake = FakeSiglipModel()
    self.vision_model = fake
    self.hidden_size = fake.config.hidden_size
    self.patch_size = fake.config.patch_size
    self.image_size = fake.config.image_size
    self.grid_size = self.image_size // self.patch_size


@pytest.fixture
def fake_dinov2():
    with patch.object(Dinov2BackboneWrapper, "__init__", _fake_dinov2_init):
        yield


@pytest.fixture
def fake_siglip():
    with patch.object(SiglipBackboneWrapper, "__init__", _fake_siglip_init):
        yield


@pytest.fixture
def fake_vjepa2():
    encoder = FakeVJepa2Encoder()
    with patch("torch.hub.load", return_value=(encoder, object())):
        yield encoder


# ── Config helpers ───────────────────────────────────────────────────

def _make_dp_config(
    vision_backbone="dinov2",
    num_cameras=1,
    state_dim=8,
    action_dim=7,
    image_size=IMAGE_SIZE,
    **overrides,
):
    input_features = {}
    for i in range(num_cameras):
        name = f"{OBS_IMAGES}.cam{i}" if num_cameras > 1 else f"{OBS_IMAGES}.image"
        input_features[name] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    defaults = dict(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        vision_backbone=vision_backbone,
        dinov2_model_name="facebook/dinov2-small" if vision_backbone == "dinov2" else None,
        siglip_model_name="google/siglip-base-patch16-224" if vision_backbone == "siglip" else None,
        device="cpu",
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        spatial_softmax_num_keypoints=16,
        down_dims=(64, 128),
        diffusion_step_embed_dim=32,
    )
    defaults.update(overrides)
    return DiffusionConfig(**defaults)


def _make_batch(config, batch_size=2):
    n_obs = config.n_obs_steps
    batch = {}
    for key, feat in config.input_features.items():
        if feat.type == FeatureType.VISUAL:
            batch[key] = torch.randn(batch_size, n_obs, *feat.shape)
        else:
            batch[key] = torch.randn(batch_size, n_obs, *feat.shape)
    batch[ACTION] = torch.randn(batch_size, config.horizon, config.action_feature.shape[0])
    batch["action_is_pad"] = torch.zeros(batch_size, config.horizon, dtype=torch.bool)
    batch[OBS_IMAGES] = torch.stack(
        [batch[key] for key in config.image_features], dim=-4
    )
    batch[OBS_STATE] = batch[OBS_STATE]
    return batch


# ══════════════════════════════════════════════════════════════════════
# Config validation tests
# ══════════════════════════════════════════════════════════════════════


class TestDiffusionConfigViT:
    def test_default_resnet_unchanged(self):
        c = DiffusionConfig()
        assert c.vision_backbone == "resnet18"
        assert c.use_group_norm is True
        assert c.crop_shape == (84, 84)

    def test_vit_auto_disables_group_norm(self):
        c = _make_dp_config(vision_backbone="dinov2")
        assert c.use_group_norm is False

    def test_vit_auto_disables_crop_shape(self):
        c = _make_dp_config(vision_backbone="dinov2")
        assert c.crop_shape is None

    def test_vit_auto_clears_pretrained_backbone_weights(self):
        c = _make_dp_config(
            vision_backbone="dinov2",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        )
        assert c.pretrained_backbone_weights is None

    def test_missing_dinov2_model_name(self):
        with pytest.raises(ValueError, match="dinov2_model_name"):
            _make_dp_config(vision_backbone="dinov2", dinov2_model_name=None)

    def test_missing_siglip_model_name(self):
        with pytest.raises(ValueError, match="siglip_model_name"):
            _make_dp_config(vision_backbone="siglip", siglip_model_name=None)

    def test_missing_mocov3_checkpoint(self):
        with pytest.raises(ValueError, match="mocov3_checkpoint_path"):
            _make_dp_config(vision_backbone="mocov3", mocov3_checkpoint_path=None)

    def test_missing_voltron_model_id(self):
        with pytest.raises(ValueError, match="voltron_model_id"):
            _make_dp_config(vision_backbone="voltron", voltron_model_id=None)

    def test_missing_cpmae_checkpoint(self):
        with pytest.raises(ValueError, match="cpmae_checkpoint_path"):
            _make_dp_config(vision_backbone="cpmae", cpmae_checkpoint_path=None)

    def test_unsupported_backbone(self):
        with pytest.raises(ValueError, match="must start with"):
            _make_dp_config(vision_backbone="vgg16")

    def test_ssl_with_vit_raises(self):
        with pytest.raises(ValueError, match="ssl_checkpoint_path"):
            _make_dp_config(
                vision_backbone="dinov2",
                ssl_checkpoint_path="/tmp/foo.pth",
            )

    def test_cpmae_non_divisible_patch_size(self):
        with pytest.raises(ValueError, match="divisible"):
            _make_dp_config(
                vision_backbone="cpmae",
                cpmae_checkpoint_path="/tmp/fake.pt",
                cpmae_img_size=224,
                cpmae_patch_size=15,
            )

    def test_vjepa2_config_valid(self):
        c = _make_dp_config(vision_backbone="vjepa2")
        assert c.vision_backbone == "vjepa2"
        assert c.vjepa2_model_name == "vjepa2_1_vit_base_384"

    def test_vjepa2_config_requires_repo(self):
        with pytest.raises(ValueError, match="vjepa2_repo_or_dir"):
            _make_dp_config(vision_backbone="vjepa2", vjepa2_repo_or_dir="")

    def test_vjepa2_config_requires_model_name(self):
        with pytest.raises(ValueError, match="vjepa2_model_name"):
            _make_dp_config(vision_backbone="vjepa2", vjepa2_model_name="")

    def test_vjepa2_config_requires_positive_input_frames(self):
        with pytest.raises(ValueError, match="vjepa2_input_frames"):
            _make_dp_config(vision_backbone="vjepa2", vjepa2_input_frames=0)

    def test_vjepa2_config_requires_positive_spatial_pool(self):
        with pytest.raises(ValueError, match="vjepa2_spatial_pool_size"):
            _make_dp_config(vision_backbone="vjepa2", vjepa2_spatial_pool_size=0)


# ══════════════════════════════════════════════════════════════════════
# V-JEPA2 backbone wrapper smoke tests
# ══════════════════════════════════════════════════════════════════════


class TestVJepa2BackboneWrapper:
    def test_wrapper_output_shape(self, fake_vjepa2):
        wrapper = VJepa2BackboneWrapper()
        x = torch.randn(2, 3, VJEPA_IMAGE, VJEPA_IMAGE)
        out = wrapper(x)
        assert out["feature_map"].shape == (2, VJEPA_HIDDEN, VJEPA_GRID, VJEPA_GRID)

    def test_wrapper_auto_resizes(self, fake_vjepa2):
        wrapper = VJepa2BackboneWrapper()
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = wrapper(x)
        assert out["feature_map"].shape == (1, VJEPA_HIDDEN, VJEPA_GRID, VJEPA_GRID)
        assert fake_vjepa2.last_input_shape == (1, 3, 1, VJEPA_IMAGE, VJEPA_IMAGE)

    def test_wrapper_averages_temporal_tokens(self, fake_vjepa2):
        wrapper = VJepa2BackboneWrapper(input_frames=2)
        x = torch.randn(1, 3, VJEPA_IMAGE, VJEPA_IMAGE)
        out = wrapper(x)
        assert out["feature_map"].shape == (1, VJEPA_HIDDEN, VJEPA_GRID, VJEPA_GRID)
        assert fake_vjepa2.last_input_shape == (1, 3, 2, VJEPA_IMAGE, VJEPA_IMAGE)

    def test_wrapper_spatial_pooling(self, fake_vjepa2):
        wrapper = VJepa2BackboneWrapper(spatial_pool_size=16)
        x = torch.randn(1, 3, VJEPA_IMAGE, VJEPA_IMAGE)
        out = wrapper(x)
        assert out["feature_map"].shape == (1, VJEPA_HIDDEN, 16, 16)


# ══════════════════════════════════════════════════════════════════════
# DiffusionRgbEncoder smoke tests
# ══════════════════════════════════════════════════════════════════════


class TestDiffusionRgbEncoderDinov2:
    def test_encoder_output_shape(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2")
        encoder = DiffusionRgbEncoder(config)
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = encoder(x)
        expected_dim = config.spatial_softmax_num_keypoints * 2
        assert out.shape == (2, expected_dim)

    def test_encoder_no_nan(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2")
        encoder = DiffusionRgbEncoder(config)
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = encoder(x)
        assert not torch.isnan(out).any()

    def test_encoder_no_crop(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2")
        encoder = DiffusionRgbEncoder(config)
        assert not encoder.do_crop

    def test_freeze_backbone(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2", freeze_backbone=True)
        encoder = DiffusionRgbEncoder(config)
        for p in encoder.backbone.parameters():
            assert not p.requires_grad


class TestDiffusionRgbEncoderSiglip:
    def test_encoder_output_shape(self, fake_siglip):
        config = _make_dp_config(vision_backbone="siglip")
        encoder = DiffusionRgbEncoder(config)
        x = torch.randn(2, 3, SIGLIP_IMAGE, SIGLIP_IMAGE)
        out = encoder(x)
        expected_dim = config.spatial_softmax_num_keypoints * 2
        assert out.shape == (2, expected_dim)

    def test_encoder_no_nan(self, fake_siglip):
        config = _make_dp_config(vision_backbone="siglip")
        encoder = DiffusionRgbEncoder(config)
        x = torch.randn(1, 3, SIGLIP_IMAGE, SIGLIP_IMAGE)
        out = encoder(x)
        assert not torch.isnan(out).any()


class TestDiffusionRgbEncoderVJepa2:
    def test_freeze_backbone(self, fake_vjepa2):
        config = _make_dp_config(vision_backbone="vjepa2", freeze_backbone=True)
        encoder = DiffusionRgbEncoder(config)
        for p in encoder.backbone.parameters():
            assert not p.requires_grad


class TestDiffusionRgbEncoderResnet:
    def test_resnet_still_works(self):
        config = _make_dp_config(
            vision_backbone="resnet18",
            image_size=96,
            crop_shape=(84, 84),
            dinov2_model_name=None,
            siglip_model_name=None,
        )
        assert config.crop_shape == (84, 84)
        encoder = DiffusionRgbEncoder(config)
        x = torch.randn(2, 3, 96, 96)
        out = encoder(x)
        expected_dim = config.spatial_softmax_num_keypoints * 2
        assert out.shape == (2, expected_dim)

    def test_resnet_with_crop(self):
        config = _make_dp_config(
            vision_backbone="resnet18",
            image_size=96,
            crop_shape=(84, 84),
            dinov2_model_name=None,
            siglip_model_name=None,
        )
        assert config.crop_shape == (84, 84)
        encoder = DiffusionRgbEncoder(config)
        assert encoder.do_crop
        x = torch.randn(2, 3, 96, 96)
        out = encoder(x)
        expected_dim = config.spatial_softmax_num_keypoints * 2
        assert out.shape == (2, expected_dim)


# ══════════════════════════════════════════════════════════════════════
# _FeatureMapDictToTensor adapter tests
# ══════════════════════════════════════════════════════════════════════


class TestFeatureMapDictToTensor:
    def test_unwraps_dict(self):
        class DummyWrapper(nn.Module):
            def forward(self, x):
                return {"feature_map": x * 2, "other": x}

        adapter = _FeatureMapDictToTensor(DummyWrapper())
        x = torch.randn(1, 3, 7, 7)
        out = adapter(x)
        torch.testing.assert_close(out, x * 2)

    def test_state_dict_nesting(self):
        inner = nn.Linear(4, 4)
        adapter = _FeatureMapDictToTensor(inner)
        sd = adapter.state_dict()
        assert any(k.startswith("inner.") for k in sd)


# ══════════════════════════════════════════════════════════════════════
# Full DiffusionPolicy forward / loss tests
# ══════════════════════════════════════════════════════════════════════


class TestDiffusionPolicyDinov2:
    def test_policy_forward_loss(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2")
        policy = DiffusionPolicy(config)
        policy.train()
        batch = _make_batch(config, batch_size=2)
        loss, out_dict = policy.forward(batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_policy_backward(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2")
        policy = DiffusionPolicy(config)
        policy.train()
        batch = _make_batch(config, batch_size=2)
        loss, _ = policy.forward(batch)
        loss.backward()
        backbone_params = [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad]
        assert len(backbone_params) > 0
        assert all(p.grad is not None for p in backbone_params)

    def test_policy_generate_actions(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2")
        policy = DiffusionPolicy(config)
        policy.eval()
        # select_action expects per-step obs (no time dim)
        obs = {key: torch.randn(1, *feat.shape) for key, feat in config.input_features.items()}
        action = policy.select_action(obs)
        assert action.shape == (1, config.action_feature.shape[0])

    def test_save_and_load(self, fake_dinov2, tmp_path):
        config = _make_dp_config(vision_backbone="dinov2")
        policy = DiffusionPolicy(config)
        policy.save_pretrained(tmp_path / "dp_dinov2")
        loaded = DiffusionPolicy.from_pretrained(tmp_path / "dp_dinov2", config=config)
        for p_orig, p_loaded in zip(policy.parameters(), loaded.parameters()):
            torch.testing.assert_close(p_orig, p_loaded, rtol=0, atol=0)

    def test_multi_camera(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2", num_cameras=2)
        policy = DiffusionPolicy(config)
        policy.train()
        batch = _make_batch(config, batch_size=2)
        loss, _ = policy.forward(batch)
        assert loss.dim() == 0

    def test_freeze_backbone_no_grad(self, fake_dinov2):
        config = _make_dp_config(vision_backbone="dinov2", freeze_backbone=True)
        policy = DiffusionPolicy(config)
        policy.train()
        batch = _make_batch(config, batch_size=2)
        loss, _ = policy.forward(batch)
        loss.backward()
        for n, p in policy.named_parameters():
            if "backbone" in n:
                assert p.grad is None, f"Expected no grad for frozen param {n}"


class TestDiffusionPolicySiglip:
    def test_policy_forward_loss(self, fake_siglip):
        config = _make_dp_config(vision_backbone="siglip")
        policy = DiffusionPolicy(config)
        policy.train()
        batch = _make_batch(config, batch_size=2)
        loss, _ = policy.forward(batch)
        assert loss.dim() == 0
        assert loss.item() > 0
