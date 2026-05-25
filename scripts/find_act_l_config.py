"""Find ACT-L transformer config that matches DP's ~257M trainable params.

Sweeps n_decoder_layers while holding dim_model, n_heads, dim_feedforward,
and n_encoder_layers fixed, printing param counts per config.
"""

from collections import defaultdict

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


TARGET_TRAINABLE = 257_400_000  # DP's trainable params with frozen R50


def _features(image_size=256, state_dim=8, action_dim=7):
    inp = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return inp, out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def group_params(policy):
    groups = defaultdict(lambda: {"total": 0, "trainable": 0})
    for name, p in policy.named_parameters():
        parts = name.split(".")
        key = parts[1] if len(parts) > 1 else parts[0]
        groups[key]["total"] += p.numel()
        if p.requires_grad:
            groups[key]["trainable"] += p.numel()
    return dict(groups)


def fmt(n):
    if n >= 1e6:
        return f"{n / 1e6:,.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:,.1f}K"
    return str(n)


def main():
    inp, out = _features(image_size=256)

    # Frozen ResNet-50 backbone (matching the DP comparison setup)
    base_config = dict(
        vision_backbone="resnet50",
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=False,
        freeze_backbone=True,
        input_features=inp,
        output_features=out,
        # Match DP's observation/action structure
        n_obs_steps=2,
        chunk_size=16,
        n_action_steps=8,
        # VAE settings
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        dropout=0.1,
        kl_weight=10.0,
        device="cpu",
    )

    # Focused sweep around the target
    candidates = [
        # d=1024 fine-tuning around 257M
        {"label": "d1024-enc4-dec9",  "dim_model": 1024, "n_heads": 16, "dim_feedforward": 4096, "n_encoder_layers": 4, "n_decoder_layers": 9},
        {"label": "d1024-enc5-dec7",  "dim_model": 1024, "n_heads": 16, "dim_feedforward": 4096, "n_encoder_layers": 5, "n_decoder_layers": 7},
        {"label": "d1024-enc5-dec8",  "dim_model": 1024, "n_heads": 16, "dim_feedforward": 4096, "n_encoder_layers": 5, "n_decoder_layers": 8},
        {"label": "d1024-enc6-dec7",  "dim_model": 1024, "n_heads": 16, "dim_feedforward": 4096, "n_encoder_layers": 6, "n_decoder_layers": 7},
        # d=512 fine-tuning (narrower but deeper)
        {"label": "d512-enc4-dec41",  "dim_model": 512, "n_heads": 8, "dim_feedforward": 3200, "n_encoder_layers": 4, "n_decoder_layers": 41},
        {"label": "d512-enc4-dec42",  "dim_model": 512, "n_heads": 8, "dim_feedforward": 3200, "n_encoder_layers": 4, "n_decoder_layers": 42},
        # d=768 wider sweep
        {"label": "d768-enc8-dec16",  "dim_model": 768, "n_heads": 12, "dim_feedforward": 3072, "n_encoder_layers": 8, "n_decoder_layers": 16},
        {"label": "d768-enc8-dec18",  "dim_model": 768, "n_heads": 12, "dim_feedforward": 3072, "n_encoder_layers": 8, "n_decoder_layers": 18},
        {"label": "d768-enc6-dec18",  "dim_model": 768, "n_heads": 12, "dim_feedforward": 3072, "n_encoder_layers": 6, "n_decoder_layers": 18},
        {"label": "d768-enc6-dec20",  "dim_model": 768, "n_heads": 12, "dim_feedforward": 3072, "n_encoder_layers": 6, "n_decoder_layers": 20},
    ]

    # Also show current default ACT for reference
    ref_config = {**base_config,
        "n_obs_steps": 1,
        "chunk_size": 100,
        "n_action_steps": 100,
        "dim_model": 512, "n_heads": 8, "dim_feedforward": 3200,
        "n_encoder_layers": 4, "n_decoder_layers": 1,
    }
    print("Building reference ACT (default)...")
    ref_policy = ACTPolicy(ACTConfig(**ref_config))
    ref_total, ref_trainable = count_params(ref_policy)
    print(f"  Reference ACT: total={fmt(ref_total)}, trainable={fmt(ref_trainable)}")
    print(f"  Target trainable: {fmt(TARGET_TRAINABLE)}")
    print()

    # Sweep candidates
    col_w = 24
    num_w = 16
    print(f"{'Config':<{col_w}} {'Total':>{num_w}} {'Trainable':>{num_w}} {'Δ Target':>{num_w}} {'Ratio':>{num_w}}")
    print("=" * (col_w + num_w * 4))

    best_label, best_delta = None, float("inf")

    for cand in candidates:
        label = cand.pop("label")
        cfg = ACTConfig(**{**base_config, **cand})
        print(f"Building {label}...", end="\r")
        policy = ACTPolicy(cfg)
        total, trainable = count_params(policy)

        delta = trainable - TARGET_TRAINABLE
        ratio = trainable / TARGET_TRAINABLE
        delta_str = f"{'+' if delta >= 0 else '-'}{fmt(abs(delta))}"
        print(f"{label:<{col_w}} {fmt(total):>{num_w}} {fmt(trainable):>{num_w}} {delta_str:>{num_w}} {ratio:>{num_w}.3f}")

        if abs(delta) < abs(best_delta):
            best_delta = delta
            best_label = label

        # Per-module breakdown for closest match
        if abs(delta / TARGET_TRAINABLE) < 0.10:
            groups = group_params(policy)
            for key, vals in sorted(groups.items()):
                print(f"    {key:<36} total={fmt(vals['total']):>12}  trainable={fmt(vals['trainable']):>12}")

        del policy

    print()
    print(f"Closest match: {best_label} (Δ = {'+' if best_delta >= 0 else '-'}{fmt(abs(best_delta))})")


if __name__ == "__main__":
    main()
