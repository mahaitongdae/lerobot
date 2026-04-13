# SSL-Pretrained ResNet Backbone Integration

## Overview

Added support for loading self-supervised learning (SSL) pretrained ResNet weights
(MoCo, SimCLR, BYOL, etc.) as vision backbones for ACT and Diffusion Policy.

## Files Changed

| File | Change |
|------|--------|
| `src/lerobot/utils/ssl_backbone.py` | **New.** Shared utility for loading SSL checkpoints with auto-detection of key prefixes. |
| `src/lerobot/policies/act/configuration_act.py` | Added `ssl_checkpoint_path` field; auto-clears `pretrained_backbone_weights` when set. |
| `src/lerobot/policies/act/modeling_act.py` | Calls `load_ssl_weights_into_resnet()` when `ssl_checkpoint_path` is set. |
| `src/lerobot/policies/diffusion/configuration_diffusion.py` | Added `ssl_checkpoint_path` field; auto-clears `pretrained_backbone_weights` and `use_group_norm` when set. |
| `src/lerobot/policies/diffusion/modeling_diffusion.py` | Calls `load_ssl_weights_into_resnet()` when `ssl_checkpoint_path` is set. |
| `tests/policies/act/test_act_ssl_backbone.py` | **New.** 43 unit tests covering utility functions, config validation, and end-to-end loading with real MoCo v2, SimCLR, and BYOL weights. |
| `tests/conftest.py` | Registered `slow` pytest marker for tests that download large checkpoints. |

## Design Decisions

### Which encoder to use per SSL method

Different SSL methods train multiple networks. We select the encoder trained with
gradient updates (not the momentum/EMA target), as it produces better-calibrated
representations for downstream fine-tuning:

| Method | Encoder selected | Key prefix stripped | Rationale |
|--------|-----------------|-------------------|-----------|
| MoCo v1/v2 | Query encoder | `module.encoder_q.` | Gradient-trained; key encoder is momentum-updated and not suitable for fine-tuning |
| MoCo v3 | Base encoder | `module.base_encoder.` | Same rationale as MoCo v2 |
| SimCLR | Backbone encoder | `backbone.` / `encoder.` | Single encoder; just need to strip wrapper prefix |
| SimCLR (VISSL) | Trunk | `_feature_blocks.` (after VISSL extraction) | VISSL nests trunk inside `classy_state_dict.base_model.model.trunk` |
| BYOL | Online encoder | `online_encoder.net.` | Gradient-trained; target encoder is EMA-updated |
| BYOL (LightlySSL) | Backbone | `backbone.` | PyTorch Lightning format |
| VISSL (general) | Trunk | `trunk._feature_blocks.` / `_feature_blocks.` | Standard VISSL convention |
| solo-learn | Backbone | `backbone.` | Standard solo-learn convention |

### Projection heads are excluded

All SSL methods train a projection head (MLP) on top of the backbone for the
contrastive/self-supervised objective. These heads are task-specific and should
**not** be loaded — only the convolutional backbone (conv1 through layer4) is
relevant for downstream visual policy learning. The loader filters out keys
containing `fc.`, `projection`, `predictor`, `head.`, or `prototypes.`.

### ACT: FrozenBatchNorm2d is compatible

ACT's ResNet backbone uses `FrozenBatchNorm2d` as `norm_layer`, which freezes
running mean/variance statistics. This is standard practice for fine-tuning and
is compatible with SSL checkpoints that were trained with regular BatchNorm —
the pretrained BN parameters are loaded and then frozen.

### Diffusion Policy: use_group_norm auto-disabled

The default Diffusion Policy replaces BatchNorm2d with GroupNorm
(`use_group_norm=True`). This is incompatible with SSL checkpoints because:
1. GroupNorm has different parameter shapes (no running_mean/running_var)
2. The replacement discards all pretrained normalization statistics

When `ssl_checkpoint_path` is set, the config automatically sets `use_group_norm=False`.

## Usage

### Common SSL checkpoint URLs

```
# MoCo v2 (800 epochs, ResNet50) — Facebook, ~375 MB
https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar

# MoCo v1 (200 epochs, ResNet50) — Facebook, ~375 MB
https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar

# SimCLR (800 epochs, ResNet50) — VISSL / Facebook, ~214 MB
https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch

# BYOL (100 epochs, ResNet50) — LightlySSL, ~410 MB
https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt
```

### ACT + SSL ResNet50

Just set `--policy.ssl_checkpoint_path` — incompatible defaults (`pretrained_backbone_weights`,
etc.) are auto-cleared by the config.

```bash
lerobot-train \
  --policy.type=act \
  --policy.vision_backbone=resnet50 \
  --policy.ssl_checkpoint_path=/path/to/moco_v2_800ep_pretrain.pth.tar \
  --policy.freeze_backbone=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  ...
```

URLs are also supported (auto-downloaded and cached by `torch.hub`):

```bash
lerobot-train \
  --policy.type=act \
  --policy.vision_backbone=resnet50 \
  --policy.ssl_checkpoint_path=https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar \
  --policy.freeze_backbone=true \
  ...
```

### Diffusion Policy + SSL ResNet50

```bash
lerobot-train \
  --policy.type=diffusion \
  --policy.vision_backbone=resnet50 \
  --policy.ssl_checkpoint_path=/path/to/simclr_resnet50.pth \
  --dataset.repo_id=HuggingFaceVLA/libero \
  ...
```

### Auto-resolved config fields

When `ssl_checkpoint_path` is set, the config automatically:
- Sets `pretrained_backbone_weights = None` (SSL weights replace ImageNet init)
- Sets `use_group_norm = False` (Diffusion only; preserves SSL BatchNorm weights)

## Verified Checkpoints

The following checkpoints are tested end-to-end in `tests/policies/act/test_act_ssl_backbone.py`
(marked `@pytest.mark.slow` since they download large files):

| Method | Source | URL | Prefix detected | Verified |
|--------|--------|-----|-----------------|----------|
| MoCo v2 (800ep) | Facebook | `https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar` | `module.encoder_q.` | bit-exact match, ACT forward, DP forward |
| SimCLR (800ep) | VISSL / Facebook | `https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch` | `_feature_blocks.` | bit-exact match, ACT forward, DP forward |
| BYOL (100ep) | LightlySSL | `https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt` | `backbone.` | bit-exact match, ACT forward, DP forward |

Run the slow tests with:

```bash
pytest tests/policies/act/test_act_ssl_backbone.py -v -m slow
```

## Auto-detection Logic

The loader (`ssl_backbone.py`) works as follows:

1. Load checkpoint; unwrap from common wrappers (`state_dict`, `model`, VISSL nested format, etc.)
2. Try each known prefix against the target ResNet's key names
3. Pick the prefix that yields the most matching keys
4. Filter out projection/prediction head parameters
5. Load with `strict=False`; log matched/missing/unexpected counts
