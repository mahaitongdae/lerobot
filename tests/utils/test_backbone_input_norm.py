#!/usr/bin/env python
"""Unit tests for per-backbone input normalization and its config integration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.backbone_input_norm import (
    PRESET_STATS,
    BackboneInputNormalizer,
    resolve_preset,
)


class _Identity(nn.Module):
    """Returns input unchanged so we can observe what the wrapper actually passes through."""

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# BackboneInputNormalizer
# ---------------------------------------------------------------------------


def test_identity_preset_is_noop():
    wrapper = BackboneInputNormalizer(_Identity(), preset="identity")
    x = torch.rand(2, 3, 8, 8)
    out = wrapper(x)
    assert torch.equal(out, x)


@pytest.mark.parametrize("preset", ["imagenet", "siglip"])
def test_known_preset_applies_expected_affine(preset: str):
    mean, std = PRESET_STATS[preset]
    wrapper = BackboneInputNormalizer(_Identity(), preset=preset)
    x = torch.rand(1, 3, 4, 4)
    out = wrapper(x)
    expected = (
        x - torch.tensor(mean).view(1, 3, 1, 1)
    ) / torch.tensor(std).view(1, 3, 1, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_custom_preset_requires_both_mean_and_std():
    with pytest.raises(ValueError, match="custom"):
        BackboneInputNormalizer(_Identity(), preset="custom", mean=(0.1, 0.2, 0.3))


def test_custom_preset_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="std"):
        BackboneInputNormalizer(
            _Identity(),
            preset="custom",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 0.0, 1.0),
        )


def test_custom_preset_applies_user_mean_std():
    wrapper = BackboneInputNormalizer(
        _Identity(),
        preset="custom",
        mean=(0.1, 0.2, 0.3),
        std=(0.5, 0.5, 0.5),
    )
    x = torch.ones(1, 3, 2, 2)
    out = wrapper(x)
    expected = (x - torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1)) / 0.5
    assert torch.allclose(out, expected, atol=1e-6)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="backbone_input_norm"):
        BackboneInputNormalizer(_Identity(), preset="banana")


def test_resolve_preset_validates_custom_length():
    with pytest.raises(ValueError, match="length 3"):
        resolve_preset("custom", mean=(0.1, 0.2), std=(0.3, 0.3, 0.3))


# ---------------------------------------------------------------------------
# Config-level validation: ACTConfig
# ---------------------------------------------------------------------------


def _act_config_kwargs(**overrides):
    base = dict(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    base.update(overrides)
    return base


def test_act_config_default_is_identity_and_preserves_visual_mean_std():
    cfg = ACTConfig(**_act_config_kwargs())
    assert cfg.backbone_input_norm == "identity"
    assert cfg.normalization_mapping["VISUAL"] == NormalizationMode.MEAN_STD


def test_act_config_imagenet_preset_forces_visual_identity():
    cfg = ACTConfig(**_act_config_kwargs(backbone_input_norm="imagenet"))
    assert cfg.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY


def test_act_config_invalid_preset_raises():
    with pytest.raises(ValueError, match="backbone_input_norm"):
        ACTConfig(**_act_config_kwargs(backbone_input_norm="banana"))


def test_act_config_custom_preset_requires_stats():
    with pytest.raises(ValueError, match="custom"):
        ACTConfig(**_act_config_kwargs(backbone_input_norm="custom"))


def test_act_config_custom_preset_roundtrip():
    cfg = ACTConfig(
        **_act_config_kwargs(
            backbone_input_norm="custom",
            backbone_input_mean=(0.1, 0.2, 0.3),
            backbone_input_std=(0.4, 0.5, 0.6),
        )
    )
    assert cfg.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY
    assert cfg.backbone_input_mean == (0.1, 0.2, 0.3)
    assert cfg.backbone_input_std == (0.4, 0.5, 0.6)


# ---------------------------------------------------------------------------
# Config-level validation: DiffusionConfig
# ---------------------------------------------------------------------------


def _dp_config_kwargs(**overrides):
    base = dict(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    base.update(overrides)
    return base


def test_dp_config_default_is_identity_and_preserves_visual_mean_std():
    cfg = DiffusionConfig(**_dp_config_kwargs())
    assert cfg.backbone_input_norm == "identity"
    assert cfg.normalization_mapping["VISUAL"] == NormalizationMode.MEAN_STD


def test_dp_config_imagenet_preset_forces_visual_identity():
    cfg = DiffusionConfig(**_dp_config_kwargs(backbone_input_norm="imagenet"))
    assert cfg.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY


def test_dp_config_invalid_preset_raises():
    with pytest.raises(ValueError, match="backbone_input_norm"):
        DiffusionConfig(**_dp_config_kwargs(backbone_input_norm="banana"))
