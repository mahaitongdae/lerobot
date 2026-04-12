"""Compare parameter counts between ACT with ResNet18, SigLIP, and DINOv2 backbones."""

from collections import defaultdict

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def _features(image_size, state_dim=8, action_dim=7):
    inp = {
        "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.state":         PolicyFeature(type=FeatureType.STATE,  shape=(state_dim,)),
    }
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return inp, out


SHARED_CONFIG = dict(
    n_obs_steps=1,
    chunk_size=100,
    n_action_steps=100,
    pre_norm=False,
    dim_model=512,
    n_heads=8,
    dim_feedforward=3200,
    feedforward_activation="relu",
    n_encoder_layers=4,
    n_decoder_layers=1,
    use_vae=True,
    latent_dim=32,
    n_vae_encoder_layers=4,
    dropout=0.1,
    kl_weight=10.0,
    device="cpu",
)


def _resnet_config():
    inp, out = _features(image_size=256)
    return {
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": None,
        "replace_final_stride_with_dilation": False,
        "input_features": inp,
        "output_features": out,
    }


def _siglip_config(model_name="google/siglip-base-patch16-224", image_size=224, freeze=False):
    inp, out = _features(image_size=image_size)
    return {
        "vision_backbone": "siglip",
        "siglip_model_name": model_name,
        "pretrained_backbone_weights": None,
        "freeze_backbone": freeze,
        "input_features": inp,
        "output_features": out,
    }


def _dinov2_config(model_name="facebook/dinov2-small", image_size=224, freeze=False):
    inp, out = _features(image_size=image_size)
    return {
        "vision_backbone": "dinov2",
        "dinov2_model_name": model_name,
        "pretrained_backbone_weights": None,
        "freeze_backbone": freeze,
        "input_features": inp,
        "output_features": out,
    }


CONFIGS = {
    "ResNet18":               _resnet_config(),
    "DINOv2-small":           _dinov2_config("facebook/dinov2-small", 224),
    "DINOv2-base":            _dinov2_config("facebook/dinov2-base", 224),
    "SigLIP-base-224":        _siglip_config("google/siglip-base-patch16-224", 224),
    "DINOv2-small frozen":    _dinov2_config("facebook/dinov2-small", 224, freeze=True),
    "SigLIP-base-224 frozen": _siglip_config("google/siglip-base-patch16-224", 224, freeze=True),
}


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
    results = {}

    for label, overrides in CONFIGS.items():
        print(f"Building {label}...")
        cfg = ACTConfig(**{**SHARED_CONFIG, **overrides})
        policy = ACTPolicy(cfg)
        total, trainable = count_params(policy)
        groups = group_params(policy)

        grid = ""
        if hasattr(policy.model, "backbone"):
            bb = policy.model.backbone
            if hasattr(bb, "grid_size"):
                grid = f"{bb.grid_size}x{bb.grid_size}"

        results[label] = {
            "total": total,
            "trainable": trainable,
            "groups": groups,
            "grid": grid,
        }

    # ── Summary table ──
    col_w = 24
    num_w = 16
    print()
    print("=" * (col_w + num_w * 3))
    print(f"{'Model':<{col_w}} {'Total':>{num_w}} {'Trainable':>{num_w}} {'Patches':>{num_w}}")
    print("-" * (col_w + num_w * 3))
    for label, r in results.items():
        grid_str = r["grid"] if r["grid"] else "8x8 (resnet)"
        print(f"{label:<{col_w}} {fmt(r['total']):>{num_w}} {fmt(r['trainable']):>{num_w}} {grid_str:>{num_w}}")
    print("=" * (col_w + num_w * 3))

    # ── Per-module breakdown ──
    all_keys = sorted(set(k for r in results.values() for k in r["groups"]))
    labels = list(results.keys())

    print(f"\n{'Module':<40}", end="")
    for label in labels:
        print(f" {label:>24}", end="")
    print()
    print("-" * (40 + 25 * len(labels)))

    for key in all_keys:
        print(f"  {key:<38}", end="")
        for label in labels:
            total = results[label]["groups"].get(key, {}).get("total", 0)
            train = results[label]["groups"].get(key, {}).get("trainable", 0)
            if total == train:
                print(f" {fmt(total):>24}", end="")
            else:
                print(f" {fmt(train):>11} / {fmt(total):<11}", end="")
        print()

    # ── Pairwise deltas vs ResNet18 ──
    base_label = "ResNet18"
    base_total = results[base_label]["total"]
    base_train = results[base_label]["trainable"]
    print(f"\n{'Comparison vs ResNet18':<40} {'Δ Total':>16} {'Δ Trainable':>16} {'× Total':>10}")
    print("-" * 82)
    for label, r in results.items():
        if label == base_label:
            continue
        dt = r["total"] - base_total
        dtr = r["trainable"] - base_train
        ratio = r["total"] / base_total
        dt_s = f"{'+' if dt >= 0 else '-'}{fmt(abs(dt))}"
        dtr_s = f"{'+' if dtr >= 0 else '-'}{fmt(abs(dtr))}"
        print(f"  {label:<38} {dt_s:>15} {dtr_s:>15} {ratio:>9.2f}x")


if __name__ == "__main__":
    main()
