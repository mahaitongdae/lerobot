# ACT-L + SD3VAE Implementation

## Problem

ACT-L with SD3 VAE backbone causes OOM on 32 GB GPUs even at batch_size=4.

**Root cause:** ACT flattens the entire spatial feature map into transformer sequence tokens
(`modeling_act.py:621/655`). SD3 VAE downsamples only 8x (224px → 28×28 = 784 tokens/camera),
whereas DINOv2 ViT-S/14 produces 16×16 = 256 tokens/camera.

With 2 cameras × 2 obs_steps:
- DINOv2: seq_len ≈ 1027 → attention storage ≈ 2.3 GiB across 12 layers
- SD3 VAE (native): seq_len ≈ 3139 → attention storage ≈ 21.1 GiB (9.3× increase from O(n²))

The softmax on a single attention matrix already tries to allocate ~6 GiB at this sequence length.

## Solution

Added **adaptive spatial pooling** in `SD3VaeBackboneWrapper` to reduce 28×28 → 14×14 before
passing features to the ACT transformer. This brings token count to 196/camera (same as ViT-S/16).

## Changes

### 1. `src/lerobot/utils/vit_backbones.py` — SD3VaeBackboneWrapper

Added `spatial_pool_size` parameter:
```python
def __init__(self, ..., spatial_pool_size: int | None = None):
    ...
    if spatial_pool_size is not None:
        self._pool = nn.AdaptiveAvgPool2d(spatial_pool_size)
    else:
        self._pool = None

def forward(self, x):
    ...
    z = (z.detach() - self._shift) * self._scaling
    if self._pool is not None:
        z = self._pool(z)  # 28x28 -> 14x14 (or custom size)
    if self.latent_proj is not None:
        z = self.latent_proj(z)
    return {"feature_map": z}
```

The pooling happens BEFORE the 1x1 projection so the projection operates on already-pooled
spatial maps (slightly more efficient, and the projection can learn from pooled statistics).

### 2. `src/lerobot/policies/act/configuration_act.py`

Added fields:
```python
sd3vae_model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers"
sd3vae_subfolder: str = "vae"
vae_latent_proj_dim: int | None = 256
vae_encode_batch_size: int = 64
vae_spatial_pool_size: int | None = 14  # 28x28 -> 14x14 by default for ACT
```

Added `"sd3vae"` to `supported_prefixes` and validation:
- Error if `sd3vae_model_name` is unset
- Warning if `freeze_backbone=True` (would freeze the trainable latent projection)
- Auto-force `backbone_input_norm="identity"` (VAE handles its own normalization)

### 3. `src/lerobot/policies/diffusion/configuration_diffusion.py`

Added `vae_spatial_pool_size: int | None = None` (default None; diffusion uses SpatialSoftmax
which already collapses spatial dims, so pooling is unnecessary).

### 4. `scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh`

- Added `SD3VAE_MODEL` variable and pre-download
- Added `sd3vae` to default backbone list
- For ACT-L + sd3vae: passes `--policy.vae_spatial_pool_size=14`
- For DP + sd3vae: no spatial pooling (SpatialSoftmax handles it)
- Sets `backbone_input_norm=identity` and `freeze_backbone=false` for sd3vae
- Batch size override: bs=16 with grad_accum=4 for ACT-L + sd3vae

## VRAM Estimate (after fix)

With `vae_spatial_pool_size=14`, ACT-L + SD3 VAE at bs=16:

| Component | VRAM |
|-----------|------|
| SD3 VAE encoder (34M, frozen) | 0.13 GB |
| ACT-L trainable + optimizer + grads | 3.9 GB |
| VAE activations (chunked, no_grad) | 0.24 GB |
| Transformer activations (seq_len≈787) | ~2 GB |
| PyTorch/CUDA overhead | ~1.5 GB |
| **Total** | **~8 GB** |

Fits comfortably on 24 GB (A5000/3090/4090) and likely on 16 GB cards.

## Usage

```bash
# Both policies, sd3vae only
BACKBONES_OVERRIDE=sd3vae bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0 1

# Dry run to verify command
DRY_RUN=1 BACKBONES_OVERRIDE=sd3vae bash scripts/cpmae/M2/vit_baselines/run_vit_sweep_joint40_dp_actl.sh 0

# Override pool size if needed (e.g., 7x7 for even fewer tokens)
# Add to command: --policy.vae_spatial_pool_size=7
```

## Design Notes

- Pool size 14 chosen to match DINOv2 ViT-S/14 token count (14×14=196), making comparisons fair
- `AdaptiveAvgPool2d` preserves spatial structure better than alternatives (strided conv, random subsample)
- The pool is placed before the 1x1 projection: this is slightly more memory-efficient and lets
  the projection learn from spatially-aggregated features
- Diffusion policy does NOT need this because it uses `SpatialSoftmax` which already reduces
  (B, C, H, W) → (B, C*2) regardless of spatial size
