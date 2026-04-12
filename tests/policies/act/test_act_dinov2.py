"""Tests for ACT policy with DINOv2 vision backbone."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, Dinov2BackboneWrapper
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


HIDDEN_SIZE = 64
PATCH_SIZE = 14
IMAGE_SIZE = 224
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 16


def _make_dinov2_config(
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
    freeze_backbone=False,
):
    input_features = {}
    for i in range(num_cameras):
        name = f"{OBS_IMAGES}.cam{i}" if num_cameras > 1 else f"{OBS_IMAGES}.image"
        input_features[name] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
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
        vision_backbone="dinov2",
        dinov2_model_name="facebook/dinov2-small",
        pretrained_backbone_weights=None,
        freeze_backbone=freeze_backbone,
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


class FakeDinov2Config:
    hidden_size = HIDDEN_SIZE
    patch_size = PATCH_SIZE
    image_size = IMAGE_SIZE


class FakeDinov2Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeDinov2Model(nn.Module):
    """Lightweight stand-in for Dinov2Model to avoid downloading weights."""

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
        # DINOv2 prepends a CLS token
        cls_token = torch.zeros(b, 1, HIDDEN_SIZE, device=hidden.device)
        hidden = torch.cat([cls_token, hidden], dim=1)
        return FakeDinov2Output(last_hidden_state=hidden)


def _fake_dinov2_init(self, model_name: str):
    nn.Module.__init__(self)
    fake = FakeDinov2Model()
    self.vision_model = fake
    self.hidden_size = fake.config.hidden_size
    self.patch_size = fake.config.patch_size
    self.image_size = fake.config.image_size
    self.grid_size = self.image_size // self.patch_size


@pytest.fixture
def fake_dinov2():
    with patch.object(Dinov2BackboneWrapper, "__init__", _fake_dinov2_init):
        yield


def _make_batch(config, batch_size=2):
    batch = {}
    for key, feat in config.input_features.items():
        batch[key] = torch.randn(batch_size, *feat.shape)
    batch[ACTION] = torch.randn(batch_size, config.chunk_size, config.action_feature.shape[0])
    batch["action_is_pad"] = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
    batch[OBS_IMAGES] = [batch[key] for key in config.image_features]
    return batch


# ── Config validation ────────────────────────────────────────────────


def test_dinov2_config_requires_model_name():
    with pytest.raises(ValueError, match="dinov2_model_name"):
        ACTConfig(
            vision_backbone="dinov2",
            dinov2_model_name=None,
            input_features={
                f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


def test_dinov2_config_valid():
    config = _make_dinov2_config()
    assert config.vision_backbone == "dinov2"
    assert config.dinov2_model_name == "facebook/dinov2-small"


# ── Dinov2BackboneWrapper ────────────────────────────────────────────


def test_dinov2_wrapper_output_shape(fake_dinov2):
    wrapper = Dinov2BackboneWrapper("facebook/dinov2-small")
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = wrapper(x)
    assert "feature_map" in out
    fm = out["feature_map"]
    assert fm.shape == (2, HIDDEN_SIZE, GRID_SIZE, GRID_SIZE)


def test_dinov2_wrapper_strips_cls_token(fake_dinov2):
    wrapper = Dinov2BackboneWrapper("facebook/dinov2-small")
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = wrapper(x)
    # CLS token stripped: num_patches = grid^2, not grid^2 + 1
    fm = out["feature_map"]
    assert fm.shape[2] * fm.shape[3] == GRID_SIZE * GRID_SIZE


def test_dinov2_wrapper_auto_resizes(fake_dinov2):
    wrapper = Dinov2BackboneWrapper("facebook/dinov2-small")
    x = torch.randn(1, 3, 256, 256)
    out = wrapper(x)
    assert out["feature_map"].shape == (1, HIDDEN_SIZE, GRID_SIZE, GRID_SIZE)


# ── ACT model with DINOv2 ───────────────────────────────────────────


def test_act_dinov2_instantiation(fake_dinov2):
    config = _make_dinov2_config()
    model = ACT(config)
    assert isinstance(model.backbone, Dinov2BackboneWrapper)


def test_act_dinov2_forward(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5)
    model = ACT(config)
    model.train()
    batch = _make_batch(config, batch_size=2)
    actions, (mu, log_sigma_x2) = model(batch)
    assert actions.shape == (2, 5, 7)
    assert mu.shape == (2, config.latent_dim)
    assert log_sigma_x2.shape == (2, config.latent_dim)


def test_act_dinov2_forward_no_vae(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5, use_vae=False)
    model = ACT(config)
    model.eval()
    batch = _make_batch(config, batch_size=2)
    del batch[ACTION]
    del batch["action_is_pad"]
    actions, (mu, log_sigma_x2) = model(batch)
    assert actions.shape == (2, 5, 7)
    assert mu is None


def test_act_dinov2_no_nan_in_output(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5)
    model = ACT(config)
    model.train()
    batch = _make_batch(config, batch_size=2)
    actions, (mu, log_sigma_x2) = model(batch)
    assert not torch.isnan(actions).any()
    assert not torch.isnan(mu).any()


# ── ACTPolicy with DINOv2 ───────────────────────────────────────────


def test_policy_forward_and_loss(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.train()
    batch = _make_batch(config, batch_size=2)
    del batch[OBS_IMAGES]
    loss, loss_dict = policy.forward(batch)
    assert loss.dim() == 0
    assert "l1_loss" in loss_dict
    assert "kld_loss" in loss_dict


def test_policy_select_action(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.eval()
    obs = {key: torch.randn(1, *feat.shape) for key, feat in config.input_features.items()}
    action = policy.select_action(obs)
    assert action.shape == (1, 7)


def test_policy_backward(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.train()
    batch = _make_batch(config, batch_size=2)
    del batch[OBS_IMAGES]
    loss, _ = policy.forward(batch)
    loss.backward()
    backbone_params = [p for n, p in policy.named_parameters() if "backbone" in n]
    assert len(backbone_params) > 0
    assert all(p.grad is not None for p in backbone_params if p.requires_grad)


def test_policy_optim_param_groups(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5)
    config.optimizer_lr_backbone = 1e-6
    policy = ACTPolicy(config)
    groups = policy.get_optim_params()
    assert len(groups) == 2
    assert groups[1]["lr"] == 1e-6
    assert len(groups[1]["params"]) > 0


# ── Freeze backbone ─────────────────────────────────────────────────


def test_freeze_backbone_disables_grads(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5, freeze_backbone=True)
    policy = ACTPolicy(config)
    for p in policy.model.backbone.parameters():
        assert not p.requires_grad


def test_freeze_backbone_excludes_from_optim(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5, freeze_backbone=True)
    policy = ACTPolicy(config)
    groups = policy.get_optim_params()
    assert len(groups[1]["params"]) == 0


def test_freeze_backbone_no_grad_after_backward(fake_dinov2):
    config = _make_dinov2_config(chunk_size=5, freeze_backbone=True)
    policy = ACTPolicy(config)
    policy.train()
    batch = _make_batch(config, batch_size=2)
    del batch[OBS_IMAGES]
    loss, _ = policy.forward(batch)
    loss.backward()
    for p in policy.model.backbone.parameters():
        assert p.grad is None


# ── Single camera ───────────────────────────────────────────────────


def test_act_dinov2_single_camera(fake_dinov2):
    config = _make_dinov2_config(num_cameras=1, chunk_size=5)
    model = ACT(config)
    model.train()
    batch = _make_batch(config, batch_size=2)
    actions, _ = model(batch)
    assert actions.shape == (2, 5, 7)


# ── Save and load ───────────────────────────────────────────────────


def test_save_and_load_pretrained(fake_dinov2, tmp_path):
    config = _make_dinov2_config(chunk_size=5)
    policy = ACTPolicy(config)
    policy.save_pretrained(tmp_path / "act_dinov2")
    loaded = ACTPolicy.from_pretrained(tmp_path / "act_dinov2", config=config)
    for p_orig, p_loaded in zip(policy.parameters(), loaded.parameters()):
        torch.testing.assert_close(p_orig, p_loaded, rtol=0, atol=0)
