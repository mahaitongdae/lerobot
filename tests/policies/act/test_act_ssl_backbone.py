"""Tests for loading MoCo v2 SSL-pretrained ResNet50 weights into ACT and Diffusion policies.

python -m pytest tests/policies/act/test_act_ssl_backbone.py
"""

import pytest
import torch
import torchvision

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.ssl_backbone import (
    _extract_state_dict,
    _filter_head_keys,
    _is_tensor_dict,
    _strip_best_prefix,
    load_ssl_weights_into_resnet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
STATE_DIM = 8
ACTION_DIM = 7


def _build_fake_moco_v2_checkpoint(tmp_path):
    """Create a fake MoCo v2 checkpoint with the standard key prefix layout."""
    model = torchvision.models.resnet50()
    sd = {}
    for k, v in model.state_dict().items():
        sd[f"module.encoder_q.{k}"] = v.clone()
    # MoCo v2 also stores the key encoder and queue — add dummy entries.
    sd["module.encoder_k.conv1.weight"] = torch.randn(64, 3, 7, 7)
    sd["module.queue"] = torch.randn(128, 65536)
    sd["module.queue_ptr"] = torch.zeros(1, dtype=torch.long)

    ckpt_path = tmp_path / "moco_v2_pretrain.pth.tar"
    torch.save({"state_dict": sd}, ckpt_path)
    return str(ckpt_path)


def _make_act_config(ssl_path, **overrides):
    defaults = dict(
        vision_backbone="resnet50",
        pretrained_backbone_weights=None,
        ssl_checkpoint_path=ssl_path,
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=16,
        use_vae=True,
        device="cpu",
        input_features={
            f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
    )
    defaults.update(overrides)
    return ACTConfig(**defaults)


def _make_diffusion_config(ssl_path, **overrides):
    defaults = dict(
        vision_backbone="resnet50",
        pretrained_backbone_weights=None,
        ssl_checkpoint_path=ssl_path,
        use_group_norm=False,
        crop_shape=(84, 84),
        device="cpu",
        input_features={
            f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
    )
    defaults.update(overrides)
    return DiffusionConfig(**defaults)


# ---------------------------------------------------------------------------
# ssl_backbone.py unit tests
# ---------------------------------------------------------------------------


class TestExtractStateDict:
    def test_unwraps_state_dict_key(self):
        inner = {"conv1.weight": torch.randn(1)}
        assert _extract_state_dict({"state_dict": inner}) is inner

    def test_unwraps_model_key(self):
        inner = {"conv1.weight": torch.randn(1)}
        assert _extract_state_dict({"model": inner}) is inner

    def test_returns_raw_if_no_wrapper(self):
        sd = {"conv1.weight": torch.randn(1)}
        assert _extract_state_dict(sd) is sd

    def test_unwraps_vissl_nested_format(self):
        trunk = {"_feature_blocks.conv1.weight": torch.randn(64, 3, 7, 7)}
        vissl_ckpt = {
            "classy_state_dict": {
                "base_model": {
                    "model": {
                        "trunk": trunk,
                        "heads": {"0.clf.0.weight": torch.randn(128, 2048)},
                    }
                }
            }
        }
        result = _extract_state_dict(vissl_ckpt)
        assert result is trunk


class TestStripBestPrefix:
    def test_moco_v2_prefix(self):
        model = torchvision.models.resnet50()
        target_keys = set(model.state_dict().keys())
        fake_sd = {f"module.encoder_q.{k}": v for k, v in model.state_dict().items()}
        stripped, prefix = _strip_best_prefix(fake_sd, target_keys)
        assert prefix == "module.encoder_q."
        assert len(set(stripped.keys()) & target_keys) == len(target_keys)

    def test_byol_prefix(self):
        model = torchvision.models.resnet50()
        target_keys = set(model.state_dict().keys())
        fake_sd = {f"online_encoder.net.{k}": v for k, v in model.state_dict().items()}
        stripped, prefix = _strip_best_prefix(fake_sd, target_keys)
        assert prefix == "online_encoder.net."

    def test_simclr_backbone_prefix(self):
        model = torchvision.models.resnet50()
        target_keys = set(model.state_dict().keys())
        fake_sd = {f"backbone.{k}": v for k, v in model.state_dict().items()}
        stripped, prefix = _strip_best_prefix(fake_sd, target_keys)
        assert prefix == "backbone."

    def test_direct_match(self):
        model = torchvision.models.resnet50()
        target_keys = set(model.state_dict().keys())
        sd = dict(model.state_dict())
        stripped, prefix = _strip_best_prefix(sd, target_keys)
        assert prefix is None

    def test_vissl_prefix(self):
        model = torchvision.models.resnet50()
        target_keys = set(model.state_dict().keys())
        fake_sd = {f"trunk.{k}": v for k, v in model.state_dict().items()}
        stripped, prefix = _strip_best_prefix(fake_sd, target_keys)
        assert prefix == "trunk."

    def test_vissl_feature_blocks_prefix(self):
        model = torchvision.models.resnet50()
        target_keys = set(model.state_dict().keys())
        fake_sd = {f"_feature_blocks.{k}": v for k, v in model.state_dict().items()}
        stripped, prefix = _strip_best_prefix(fake_sd, target_keys)
        assert prefix == "_feature_blocks."


class TestFilterHeadKeys:
    def test_removes_fc_keys(self):
        sd = {"conv1.weight": torch.randn(1), "fc.weight": torch.randn(1), "fc.bias": torch.randn(1)}
        filtered = _filter_head_keys(sd)
        assert "fc.weight" not in filtered
        assert "conv1.weight" in filtered

    def test_removes_projection_keys(self):
        sd = {"layer1.0.conv1.weight": torch.randn(1), "projection.0.weight": torch.randn(1)}
        filtered = _filter_head_keys(sd)
        assert "projection.0.weight" not in filtered


class TestLoadSslWeightsIntoResnet:
    def test_loads_moco_v2_checkpoint(self, tmp_path):
        ckpt_path = _build_fake_moco_v2_checkpoint(tmp_path)
        model = torchvision.models.resnet50()
        before = model.layer4[0].conv1.weight.clone()

        load_ssl_weights_into_resnet(model, ckpt_path)

        # Weights should have changed (since the fake checkpoint has different random weights)
        after = model.layer4[0].conv1.weight
        assert not torch.equal(before, after)

    def test_missing_file_raises(self):
        model = torchvision.models.resnet50()
        with pytest.raises(FileNotFoundError, match="SSL checkpoint not found"):
            load_ssl_weights_into_resnet(model, "/nonexistent/path/to/ckpt.pth")

    def test_unmatched_checkpoint_raises(self, tmp_path):
        bad_sd = {"totally.wrong.key": torch.randn(1)}
        ckpt_path = tmp_path / "bad.pth"
        torch.save(bad_sd, ckpt_path)
        model = torchvision.models.resnet50()
        with pytest.raises(RuntimeError, match="Could not match any SSL checkpoint keys"):
            load_ssl_weights_into_resnet(model, str(ckpt_path))

    def test_loaded_weights_match_source(self, tmp_path):
        """Verify that after loading, backbone weights are byte-identical to the source."""
        source_model = torchvision.models.resnet50()
        sd = {f"module.encoder_q.{k}": v.clone() for k, v in source_model.state_dict().items()}
        ckpt_path = tmp_path / "moco.pth.tar"
        torch.save({"state_dict": sd}, ckpt_path)

        target_model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(target_model, str(ckpt_path))

        for key in source_model.state_dict():
            if "fc." in key:
                continue
            torch.testing.assert_close(
                target_model.state_dict()[key],
                source_model.state_dict()[key],
                rtol=0,
                atol=0,
            )


# ---------------------------------------------------------------------------
# ACTConfig validation
# ---------------------------------------------------------------------------


class TestACTConfigSSL:
    def test_ssl_auto_clears_pretrained_weights(self):
        config = _make_act_config(
            ssl_path="/tmp/ckpt.pth",
            pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        assert config.pretrained_backbone_weights is None

    def test_ssl_with_non_resnet_raises(self):
        with pytest.raises(ValueError, match="only supported for ResNet"):
            ACTConfig(
                vision_backbone="siglip",
                siglip_model_name="google/siglip-base-patch16-224",
                pretrained_backbone_weights=None,
                ssl_checkpoint_path="/tmp/ckpt.pth",
                device="cpu",
                input_features={
                    f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
                },
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
            )

    def test_ssl_resnet50_config_valid(self):
        config = _make_act_config(ssl_path="/tmp/ckpt.pth")
        assert config.ssl_checkpoint_path == "/tmp/ckpt.pth"
        assert config.pretrained_backbone_weights is None


# ---------------------------------------------------------------------------
# DiffusionConfig validation
# ---------------------------------------------------------------------------


class TestDiffusionConfigSSL:
    def test_ssl_auto_disables_group_norm(self):
        config = _make_diffusion_config(ssl_path="/tmp/ckpt.pth", use_group_norm=True)
        assert not config.use_group_norm

    def test_ssl_auto_clears_pretrained_weights(self):
        config = _make_diffusion_config(
            ssl_path="/tmp/ckpt.pth",
            pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V1",
            use_group_norm=False,
        )
        assert config.pretrained_backbone_weights is None

    def test_ssl_resnet50_config_valid(self):
        config = _make_diffusion_config(ssl_path="/tmp/ckpt.pth")
        assert config.ssl_checkpoint_path == "/tmp/ckpt.pth"
        assert not config.use_group_norm


# ---------------------------------------------------------------------------
# End-to-end: ACT model with MoCo v2 checkpoint
# ---------------------------------------------------------------------------


class TestACTWithMoCoV2:
    def test_model_instantiation(self, tmp_path):
        ckpt_path = _build_fake_moco_v2_checkpoint(tmp_path)
        config = _make_act_config(ssl_path=ckpt_path)
        model = ACT(config)
        assert model.backbone is not None

    def test_forward_pass(self, tmp_path):
        ckpt_path = _build_fake_moco_v2_checkpoint(tmp_path)
        config = _make_act_config(ssl_path=ckpt_path, chunk_size=5, n_action_steps=5)
        model = ACT(config)
        model.train()

        batch = {
            OBS_STATE: torch.randn(2, STATE_DIM),
            ACTION: torch.randn(2, 5, ACTION_DIM),
            "action_is_pad": torch.zeros(2, 5, dtype=torch.bool),
            OBS_IMAGES: [torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)],
        }
        actions, (mu, log_sigma_x2) = model(batch)
        assert actions.shape == (2, 5, ACTION_DIM)
        assert not torch.isnan(actions).any()

    def test_backbone_weights_differ_from_random(self, tmp_path):
        """After loading MoCo v2, backbone should differ from a fresh random ResNet50."""
        ckpt_path = _build_fake_moco_v2_checkpoint(tmp_path)
        config_ssl = _make_act_config(ssl_path=ckpt_path)
        config_rand = _make_act_config(ssl_path=None)

        torch.manual_seed(42)
        model_ssl = ACT(config_ssl)
        torch.manual_seed(42)
        model_rand = ACT(config_rand)

        ssl_param = next(model_ssl.backbone.parameters())
        rand_param = next(model_rand.backbone.parameters())
        assert not torch.equal(ssl_param, rand_param)


# ---------------------------------------------------------------------------
# End-to-end: DiffusionRgbEncoder with MoCo v2 checkpoint
# ---------------------------------------------------------------------------


class TestDiffusionRgbEncoderWithMoCoV2:
    def test_encoder_instantiation(self, tmp_path):
        ckpt_path = _build_fake_moco_v2_checkpoint(tmp_path)
        config = _make_diffusion_config(ssl_path=ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        assert encoder.backbone is not None

    def test_forward_pass(self, tmp_path):
        ckpt_path = _build_fake_moco_v2_checkpoint(tmp_path)
        config = _make_diffusion_config(ssl_path=ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = encoder(x)
        assert out.shape[0] == 2
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# End-to-end with real MoCo v2 weights (downloads ~375 MB on first run)
# ---------------------------------------------------------------------------

MOCO_V2_URL = (
    "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
    "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
)


@pytest.mark.slow
class TestRealMoCoV2Weights:
    """Tests that download and load the official MoCo v2 800-epoch checkpoint."""

    @pytest.fixture(autouse=True, scope="class")
    def moco_v2_checkpoint(self, tmp_path_factory):
        """Download once per test class, reuse across methods."""
        cache_dir = tmp_path_factory.mktemp("moco_v2_cache")
        ckpt_path = cache_dir / "moco_v2_800ep_pretrain.pth.tar"
        state_dict = torch.hub.load_state_dict_from_url(
            MOCO_V2_URL, model_dir=str(cache_dir), map_location="cpu"
        )
        torch.save(state_dict, ckpt_path)
        self.__class__._ckpt_path = str(ckpt_path)
        self.__class__._raw_sd = state_dict["state_dict"]

    @property
    def ckpt_path(self):
        return self.__class__._ckpt_path

    @property
    def raw_sd(self):
        return self.__class__._raw_sd

    def test_prefix_detection(self):
        target_keys = set(torchvision.models.resnet50().state_dict().keys())
        stripped, prefix = _strip_best_prefix(self.raw_sd, target_keys)
        assert prefix == "module.encoder_q."
        backbone_keys = {k for k in stripped if k in target_keys}
        # ResNet50 has 320 parameters; fc.weight + fc.bias = 2 that may be included
        assert len(backbone_keys) >= 318

    def test_load_into_resnet50(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)
        # conv1.weight should no longer be the default init
        # (MoCo trains from scratch so weights differ from both random init and ImageNet)
        assert model.conv1.weight.abs().mean() > 0

    def test_loaded_weights_match_source_exactly(self):
        """Every non-fc backbone key must be bit-identical to the checkpoint."""
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)

        source = {
            k.replace("module.encoder_q.", ""): v
            for k, v in self.raw_sd.items()
            if k.startswith("module.encoder_q.")
        }
        for key, param in model.state_dict().items():
            if "fc." in key or key not in source:
                continue
            torch.testing.assert_close(param, source[key], rtol=0, atol=0)

    def test_act_forward_with_real_weights(self):
        config = _make_act_config(ssl_path=self.ckpt_path, chunk_size=5, n_action_steps=5)
        model = ACT(config)
        model.eval()
        batch = {
            OBS_STATE: torch.randn(1, STATE_DIM),
            OBS_IMAGES: [torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)],
        }
        with torch.no_grad():
            actions, (mu, log_sigma_x2) = model(batch)
        assert actions.shape == (1, 5, ACTION_DIM)
        assert not torch.isnan(actions).any()

    def test_diffusion_encoder_forward_with_real_weights(self):
        config = _make_diffusion_config(ssl_path=self.ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        encoder.eval()
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape[0] == 1
        assert not torch.isnan(out).any()

    def test_url_loading_directly(self):
        """Load from URL instead of local file path."""
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, MOCO_V2_URL)
        assert model.conv1.weight.abs().mean() > 0


# ---------------------------------------------------------------------------
# End-to-end with real SimCLR weights (VISSL, 800 epochs, ~214 MB)
# ---------------------------------------------------------------------------

SIMCLR_VISSL_URL = (
    "https://dl.fbaipublicfiles.com/vissl/model_zoo/"
    "simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/"
    "model_final_checkpoint_phase799.torch"
)


@pytest.mark.slow
class TestRealSimCLRWeights:
    """Tests that download and load the VISSL SimCLR 800-epoch ResNet50 checkpoint."""

    @pytest.fixture(autouse=True, scope="class")
    def simclr_checkpoint(self, tmp_path_factory):
        cache_dir = tmp_path_factory.mktemp("simclr_cache")
        ckpt_path = cache_dir / "simclr_rn50_800ep.torch"
        ckpt = torch.hub.load_state_dict_from_url(
            SIMCLR_VISSL_URL, model_dir=str(cache_dir), map_location="cpu"
        )
        torch.save(ckpt, ckpt_path)
        self.__class__._ckpt_path = str(ckpt_path)
        # Extract the trunk state_dict for direct comparison
        self.__class__._trunk_sd = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

    @property
    def ckpt_path(self):
        return self.__class__._ckpt_path

    @property
    def trunk_sd(self):
        return self.__class__._trunk_sd

    def test_vissl_extraction_and_prefix(self):
        target_keys = set(torchvision.models.resnet50().state_dict().keys())
        stripped, prefix = _strip_best_prefix(self.trunk_sd, target_keys)
        assert prefix == "_feature_blocks."
        assert len(set(stripped.keys()) & target_keys) >= 318

    def test_load_into_resnet50(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)
        assert model.conv1.weight.abs().mean() > 0

    def test_loaded_weights_match_source_exactly(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)

        source = {
            k.replace("_feature_blocks.", ""): v
            for k, v in self.trunk_sd.items()
            if k.startswith("_feature_blocks.")
        }
        for key, param in model.state_dict().items():
            if "fc." in key or key not in source:
                continue
            torch.testing.assert_close(param, source[key], rtol=0, atol=0)

    def test_act_forward_with_simclr(self):
        config = _make_act_config(ssl_path=self.ckpt_path, chunk_size=5, n_action_steps=5)
        model = ACT(config)
        model.eval()
        batch = {
            OBS_STATE: torch.randn(1, STATE_DIM),
            OBS_IMAGES: [torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)],
        }
        with torch.no_grad():
            actions, _ = model(batch)
        assert actions.shape == (1, 5, ACTION_DIM)
        assert not torch.isnan(actions).any()

    def test_diffusion_encoder_forward_with_simclr(self):
        config = _make_diffusion_config(ssl_path=self.ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out.shape[0] == 1
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# End-to-end with real BYOL weights (LightlySSL, 100 epochs, ~410 MB)
# ---------------------------------------------------------------------------

BYOL_LIGHTLY_URL = (
    "https://lightly-ssl-checkpoints.s3.amazonaws.com/"
    "imagenet_resnet50_byol_2024-02-14_16-10-09/"
    "pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt"
)


@pytest.mark.slow
class TestRealBYOLWeights:
    """Tests that download and load the LightlySSL BYOL 100-epoch ResNet50 checkpoint."""

    @pytest.fixture(autouse=True, scope="class")
    def byol_checkpoint(self, tmp_path_factory):
        cache_dir = tmp_path_factory.mktemp("byol_cache")
        ckpt_path = cache_dir / "byol_rn50_100ep.ckpt"
        ckpt = torch.hub.load_state_dict_from_url(
            BYOL_LIGHTLY_URL, model_dir=str(cache_dir), map_location="cpu"
        )
        torch.save(ckpt, ckpt_path)
        self.__class__._ckpt_path = str(ckpt_path)
        self.__class__._raw_sd = ckpt["state_dict"]

    @property
    def ckpt_path(self):
        return self.__class__._ckpt_path

    @property
    def raw_sd(self):
        return self.__class__._raw_sd

    def test_prefix_detection(self):
        target_keys = set(torchvision.models.resnet50().state_dict().keys())
        stripped, prefix = _strip_best_prefix(self.raw_sd, target_keys)
        assert prefix == "backbone."
        assert len(set(stripped.keys()) & target_keys) >= 318

    def test_load_into_resnet50(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)
        assert model.conv1.weight.abs().mean() > 0

    def test_loaded_weights_match_source_exactly(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)

        source = {
            k.replace("backbone.", ""): v
            for k, v in self.raw_sd.items()
            if k.startswith("backbone.")
        }
        for key, param in model.state_dict().items():
            if "fc." in key or key not in source:
                continue
            torch.testing.assert_close(param, source[key], rtol=0, atol=0)

    def test_act_forward_with_byol(self):
        config = _make_act_config(ssl_path=self.ckpt_path, chunk_size=5, n_action_steps=5)
        model = ACT(config)
        model.eval()
        batch = {
            OBS_STATE: torch.randn(1, STATE_DIM),
            OBS_IMAGES: [torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)],
        }
        with torch.no_grad():
            actions, _ = model(batch)
        assert actions.shape == (1, 5, ACTION_DIM)
        assert not torch.isnan(actions).any()

    def test_diffusion_encoder_forward_with_byol(self):
        config = _make_diffusion_config(ssl_path=self.ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out.shape[0] == 1
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# End-to-end with real VIP weights (Ma et al., ICLR 2023; direct S3, ~94 MB)
# ---------------------------------------------------------------------------

VIP_URL = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"


@pytest.mark.slow
class TestRealVIPWeights:
    """Tests that download and load the official VIP ResNet-50 checkpoint.

    VIP saves the DataParallel-wrapped model under a top-level "vip" key, with the
    ResNet-50 backbone living under the "module.convnet." prefix.
    """

    @pytest.fixture(autouse=True, scope="class")
    def vip_checkpoint(self, tmp_path_factory):
        cache_dir = tmp_path_factory.mktemp("vip_cache")
        ckpt_path = cache_dir / "vip_model.pt"
        ckpt = torch.hub.load_state_dict_from_url(
            VIP_URL, model_dir=str(cache_dir), map_location="cpu"
        )
        torch.save(ckpt, ckpt_path)
        self.__class__._ckpt_path = str(ckpt_path)
        self.__class__._raw_sd = ckpt["vip"]

    @property
    def ckpt_path(self):
        return self.__class__._ckpt_path

    @property
    def raw_sd(self):
        return self.__class__._raw_sd

    def test_wrapper_key_extraction(self):
        """`_extract_state_dict` should unwrap the top-level 'vip' key."""
        extracted = _extract_state_dict({"vip": self.raw_sd, "optim": {}})
        assert extracted is self.raw_sd

    def test_prefix_detection(self):
        target_keys = set(torchvision.models.resnet50().state_dict().keys())
        stripped, prefix = _strip_best_prefix(self.raw_sd, target_keys)
        assert prefix == "module.convnet."
        assert len(set(stripped.keys()) & target_keys) >= 318

    def test_load_into_resnet50(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)
        assert model.conv1.weight.abs().mean() > 0

    def test_loaded_weights_match_source_exactly(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)

        source = {
            k.replace("module.convnet.", ""): v
            for k, v in self.raw_sd.items()
            if k.startswith("module.convnet.")
        }
        for key, param in model.state_dict().items():
            if "fc." in key or key not in source:
                continue
            torch.testing.assert_close(param, source[key], rtol=0, atol=0)

    def test_act_forward_with_vip(self):
        config = _make_act_config(ssl_path=self.ckpt_path, chunk_size=5, n_action_steps=5)
        model = ACT(config)
        model.eval()
        batch = {
            OBS_STATE: torch.randn(1, STATE_DIM),
            OBS_IMAGES: [torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)],
        }
        with torch.no_grad():
            actions, _ = model(batch)
        assert actions.shape == (1, 5, ACTION_DIM)
        assert not torch.isnan(actions).any()

    def test_diffusion_encoder_forward_with_vip(self):
        config = _make_diffusion_config(ssl_path=self.ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out.shape[0] == 1
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# End-to-end with real R3M weights (Nair et al., CoRL 2022; via `r3m` pip package)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealR3MWeights:
    """Tests that download and load the official R3M ResNet-50 checkpoint via gdown.

    R3M saves the DataParallel-wrapped model under a top-level "r3m" key, with the
    ResNet-50 backbone living under the "module.convnet." prefix (same layout as VIP).
    Requires: `pip install git+https://github.com/facebookresearch/r3m`
    """

    @pytest.fixture(autouse=True, scope="class")
    def r3m_checkpoint(self):
        import os
        from os.path import expanduser

        r3m = pytest.importorskip(
            "r3m", reason="r3m package not installed (pip install git+https://github.com/facebookresearch/r3m)"
        )
        # Triggers gdown download on first run; subsequent calls are cached.
        r3m.load_r3m("resnet50")
        ckpt_path = os.path.join(expanduser("~"), ".r3m", "r3m_50", "model.pt")
        assert os.path.isfile(ckpt_path), f"R3M checkpoint not found at {ckpt_path}"
        self.__class__._ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.__class__._raw_sd = ckpt["r3m"]

    @property
    def ckpt_path(self):
        return self.__class__._ckpt_path

    @property
    def raw_sd(self):
        return self.__class__._raw_sd

    def test_wrapper_key_extraction(self):
        extracted = _extract_state_dict({"r3m": self.raw_sd, "optim": {}})
        assert extracted is self.raw_sd

    def test_prefix_detection(self):
        target_keys = set(torchvision.models.resnet50().state_dict().keys())
        stripped, prefix = _strip_best_prefix(self.raw_sd, target_keys)
        assert prefix == "module.convnet."
        assert len(set(stripped.keys()) & target_keys) >= 318

    def test_load_into_resnet50(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)
        assert model.conv1.weight.abs().mean() > 0

    def test_loaded_weights_match_source_exactly(self):
        model = torchvision.models.resnet50()
        load_ssl_weights_into_resnet(model, self.ckpt_path)

        source = {
            k.replace("module.convnet.", ""): v
            for k, v in self.raw_sd.items()
            if k.startswith("module.convnet.")
        }
        for key, param in model.state_dict().items():
            if "fc." in key or key not in source:
                continue
            torch.testing.assert_close(param, source[key], rtol=0, atol=0)

    def test_act_forward_with_r3m(self):
        config = _make_act_config(ssl_path=self.ckpt_path, chunk_size=5, n_action_steps=5)
        model = ACT(config)
        model.eval()
        batch = {
            OBS_STATE: torch.randn(1, STATE_DIM),
            OBS_IMAGES: [torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)],
        }
        with torch.no_grad():
            actions, _ = model(batch)
        assert actions.shape == (1, 5, ACTION_DIM)
        assert not torch.isnan(actions).any()

    def test_diffusion_encoder_forward_with_r3m(self):
        config = _make_diffusion_config(ssl_path=self.ckpt_path)
        encoder = DiffusionRgbEncoder(config)
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out.shape[0] == 1
        assert not torch.isnan(out).any()
