"""Integration test for ACT-L: multi-obs forward pass + param count verification."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


ACT_L_CONFIG = dict(
    # Match DP observation/action structure
    n_obs_steps=2,
    chunk_size=16,
    n_action_steps=8,
    # Scaled transformer (d768-enc8-dec18: 256.88M trainable ≈ DP's 257.4M)
    dim_model=768,
    n_heads=12,
    dim_feedforward=3072,
    feedforward_activation="relu",
    n_encoder_layers=8,
    n_decoder_layers=18,
    # VAE
    use_vae=True,
    latent_dim=32,
    n_vae_encoder_layers=4,
    # Frozen ResNet-50 backbone (same as DP comparison)
    vision_backbone="resnet50",
    pretrained_backbone_weights=None,
    replace_final_stride_with_dilation=False,
    freeze_backbone=True,
    # Other
    dropout=0.1,
    kl_weight=10.0,
    device="cpu",
)


def _features(image_size=256, state_dim=8, action_dim=7):
    inp = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return inp, out


def test_config_properties():
    """Verify observation_delta_indices and action_delta_indices for ACT-L."""
    inp, out = _features()
    cfg = ACTConfig(**ACT_L_CONFIG, input_features=inp, output_features=out)

    assert cfg.n_obs_steps == 2
    assert cfg.observation_delta_indices == [-1, 0], (
        f"Expected [-1, 0], got {cfg.observation_delta_indices}"
    )
    assert cfg.action_delta_indices == list(range(-1, -1 + 16)), (
        f"Expected [-1, 0, ..., 14], got {cfg.action_delta_indices}"
    )
    print("[PASS] Config properties correct")

    # Verify backward compatibility: default ACT config
    default_cfg = ACTConfig(input_features=inp, output_features=out, device="cpu")
    assert default_cfg.n_obs_steps == 1
    assert default_cfg.observation_delta_indices is None
    assert default_cfg.action_delta_indices == list(range(100))
    print("[PASS] Default ACT backward compatibility preserved")


def test_forward_pass_training():
    """Test ACT-L forward pass in training mode with multi-obs batch."""
    inp, out = _features(image_size=256, state_dim=8, action_dim=7)
    cfg = ACTConfig(**ACT_L_CONFIG, input_features=inp, output_features=out)
    policy = ACTPolicy(cfg)
    policy.train()

    B, T, S, A = 2, 2, 8, 7

    batch = {
        "observation.state": torch.randn(B, T, S),
        "observation.images.image": torch.randn(B, T, 3, 256, 256),
        "observation.images.image2": torch.randn(B, T, 3, 256, 256),
        "action": torch.randn(B, 16, A),
        "action_is_pad": torch.zeros(B, 16, dtype=torch.bool),
    }

    loss, output_dict = policy.forward(batch)
    print(f"[PASS] Training forward pass: loss = {loss.item():.4f}")


def test_forward_pass_inference():
    """Test ACT-L forward pass in inference mode with multi-obs batch."""
    inp, out = _features(image_size=256, state_dim=8, action_dim=7)
    cfg = ACTConfig(**ACT_L_CONFIG, input_features=inp, output_features=out)
    policy = ACTPolicy(cfg)
    policy.eval()

    B, T, A = 1, 2, 7

    batch = {
        "observation.state": torch.randn(B, T, 8),
        "observation.images.image": torch.randn(B, T, 3, 256, 256),
        "observation.images.image2": torch.randn(B, T, 3, 256, 256),
    }

    actions = policy.predict_action_chunk(batch)
    assert actions.shape == (B, 16, A), f"Expected (1, 16, 7), got {actions.shape}"
    print(f"[PASS] Inference forward pass: actions shape = {actions.shape}")


def test_select_action_queue():
    """Test ACT-L select_action with observation queuing for multi-obs inference."""
    inp, out = _features(image_size=256, state_dim=8, action_dim=7)
    cfg = ACTConfig(**ACT_L_CONFIG, input_features=inp, output_features=out)
    policy = ACTPolicy(cfg)
    policy.eval()
    policy.reset()

    B, A = 1, 7

    # Simulate single-step observations (as env would provide)
    for step in range(16):
        batch = {
            "observation.state": torch.randn(B, 8),
            "observation.images.image": torch.randn(B, 3, 256, 256),
            "observation.images.image2": torch.randn(B, 3, 256, 256),
        }
        action = policy.select_action(batch)
        assert action.shape == (B, A), f"Step {step}: expected ({B}, {A}), got {action.shape}"

    print("[PASS] select_action queue works for 16 steps")


def test_default_act_unchanged():
    """Verify that default ACT (n_obs_steps=1) still works identically."""
    inp, out = _features(image_size=256, state_dim=8, action_dim=7)
    cfg = ACTConfig(
        input_features=inp, output_features=out,
        vision_backbone="resnet18", pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    policy.train()

    B, S, A = 2, 8, 7

    batch = {
        "observation.state": torch.randn(B, S),
        "observation.images.image": torch.randn(B, 3, 256, 256),
        "observation.images.image2": torch.randn(B, 3, 256, 256),
        "action": torch.randn(B, 100, A),
        "action_is_pad": torch.zeros(B, 100, dtype=torch.bool),
    }

    loss, _ = policy.forward(batch)
    print(f"[PASS] Default ACT still works: loss = {loss.item():.4f}")


def test_param_count():
    """Verify ACT-L trainable params match DP target."""
    inp, out = _features()
    cfg = ACTConfig(**ACT_L_CONFIG, input_features=inp, output_features=out)
    policy = ACTPolicy(cfg)

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    target = 257_400_000
    ratio = trainable / target
    print(f"[INFO] ACT-L: total={total:,}, trainable={trainable:,} (ratio={ratio:.3f} of DP target)")
    assert 0.95 <= ratio <= 1.05, f"Trainable params {trainable:,} not within 5% of target {target:,}"
    print("[PASS] Param count within 5% of DP target")


if __name__ == "__main__":
    test_config_properties()
    test_param_count()
    test_default_act_unchanged()
    test_forward_pass_training()
    test_forward_pass_inference()
    test_select_action_queue()
    print("\n=== All ACT-L integration tests passed ===")
