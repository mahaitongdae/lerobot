# ACT vs Diffusion ResNet Encoder Comparison

This note compares the structure of the visual pipeline in:

- `src/lerobot/policies/act/modeling_act.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`

The emphasis is on:

- what feature each model extracts from the ResNet encoder
- how that feature is transformed after extraction
- how the downstream policy consumes the resulting representation

## Executive Summary

Both models use the last convolutional stage of a ResNet, before global average pooling and the classifier head.

- ACT keeps the spatial feature map as a dense 2D grid and turns every spatial location into a transformer token.
- Diffusion compresses the spatial feature map into a small set of keypoints with `SpatialSoftmax`, then into a single per-image vector used only as global conditioning.

In short:

- ACT uses ResNet output as a spatial token sequence.
- Diffusion uses ResNet output as a compact global descriptor.

## Common Ground

Both models stop the ResNet before the classification head.

- ACT explicitly asks for `layer4` via `IntermediateLayerGetter`:
  `src/lerobot/policies/act/modeling_act.py:401-410`
- Diffusion keeps `list(backbone_model.children())[:-2]`, which is also the backbone up to the last conv feature map:
  `src/lerobot/policies/diffusion/modeling_diffusion.py:460-466`

So in both cases the extracted visual feature is the final convolutional feature map:

- shape conceptually: `(B, C, H', W')`

The major difference is what happens next.

## ACT Path

### Backbone output

For the default ResNet path, ACT builds:

- `torchvision.models.resnet18(...)`
- `norm_layer=FrozenBatchNorm2d`
- `IntermediateLayerGetter(..., return_layers={"layer4": "feature_map"})`

Reference:

- `src/lerobot/policies/act/modeling_act.py:401-410`

The output consumed by ACT is:

- `feature_map = self.backbone(img)["feature_map"]`
- shape: `(B, C, H', W')`

Reference:

- `src/lerobot/policies/act/modeling_act.py:553-556`

### Post-processing of the ResNet feature map

ACT does not collapse the spatial map into a single vector. Instead it preserves the grid structure.

Processing steps:

1. Apply a `1x1` convolution to project channels from `backbone_out_channels` to `dim_model`.
2. Generate a 2D sinusoidal positional embedding on the same `(H', W')` grid.
3. Rearrange the projected feature map from `(B, D, H', W')` to `((H' * W'), B, D)`.
4. Treat every spatial position as a transformer token.

References:

- image projection: `src/lerobot/policies/act/modeling_act.py:430-432`
- 2D position embedding module: `src/lerobot/policies/act/modeling_act.py:441-442`
- reshaping into tokens: `src/lerobot/policies/act/modeling_act.py:553-565`
- 2D sinusoidal embedding definition: `src/lerobot/policies/act/modeling_act.py:765-816`

### How ACT uses the visual tokens downstream

ACT builds one long encoder sequence consisting of:

- latent token
- optional robot-state token
- optional env-state token
- all image spatial tokens from all cameras

References:

- token structure comment: `src/lerobot/policies/act/modeling_act.py:419-420`
- encoder input assembly: `src/lerobot/policies/act/modeling_act.py:539-569`

Those tokens are then passed through a transformer encoder:

- `encoder_out = self.encoder(...)`

Reference:

- `src/lerobot/policies/act/modeling_act.py:571-572`

Then a transformer decoder uses the encoded memory to predict the action chunk:

- `decoder_out = self.decoder(...)`

Reference:

- `src/lerobot/policies/act/modeling_act.py:573-583`

### Practical interpretation

ACT treats the ResNet feature map as a structured visual scene representation. The model can attend over individual spatial locations and combine them with state tokens inside the transformer.

The main consequence is:

- visual spatial layout is preserved deep into the policy

## Diffusion Path

### Backbone output

Diffusion builds:

- `torchvision.models.resnet18(...)` or another configured ResNet
- a sequential trunk using all layers except avgpool and fc

Reference:

- `src/lerobot/policies/diffusion/modeling_diffusion.py:460-466`

The raw backbone output is also a final conv feature map:

- shape conceptually: `(B, C, H', W')`

### Post-processing of the ResNet feature map

Diffusion immediately compresses the spatial map into a fixed-length vector.

Processing steps:

1. Optionally crop the input image.
2. Run the ResNet trunk to get `(B, C, H', W')`.
3. Apply `SpatialSoftmax` to convert the feature map into `num_kp` image-space keypoints.
4. Flatten the keypoints from `(B, K, 2)` to `(B, 2K)`.
5. Apply `Linear(2K, 2K)` followed by `ReLU`.

References:

- crop setup: `src/lerobot/policies/diffusion/modeling_diffusion.py:448-458`
- optional BN -> GN replacement: `src/lerobot/policies/diffusion/modeling_diffusion.py:467-476`
- spatial softmax setup: `src/lerobot/policies/diffusion/modeling_diffusion.py:478-493`
- forward path: `src/lerobot/policies/diffusion/modeling_diffusion.py:495-513`
- `SpatialSoftmax` definition: `src/lerobot/policies/diffusion/modeling_diffusion.py:369-437`

This means the extracted image representation is not a token grid. It is a compact vector:

- `feature_dim = spatial_softmax_num_keypoints * 2`

Reference:

- `src/lerobot/policies/diffusion/modeling_diffusion.py:490-492`

### How Diffusion uses the image vector downstream

Diffusion concatenates:

- robot state across observation steps
- image vectors across cameras and observation steps
- optional env state across observation steps

Then it flattens everything into one `global_cond` vector per batch element.

References:

- global conditioning assembly: `src/lerobot/policies/diffusion/modeling_diffusion.py:238-274`
- use in action generation: `src/lerobot/policies/diffusion/modeling_diffusion.py:290-299`

That `global_cond` vector is not used as a token sequence. Instead it conditions a 1D temporal U-Net that denoises an action trajectory.

References:

- U-Net construction: `src/lerobot/policies/diffusion/modeling_diffusion.py:593-662`
- conditioning injected via FiLM-style residual blocks: `src/lerobot/policies/diffusion/modeling_diffusion.py:677-705`
- FiLM conditioning block: `src/lerobot/policies/diffusion/modeling_diffusion.py:709-760`

### Practical interpretation

Diffusion treats the ResNet output as an observation summary used to condition the action denoising process. Spatial detail is compressed before the policy core sees it.

The main consequence is:

- visual spatial layout is mostly summarized away before temporal modeling

## Key Structural Differences

### 1. What is extracted from ResNet

Both start from the last conv feature map, but they retain different structure.

- ACT retains the full `(H', W')` grid as transformer tokens.
- Diffusion reduces the grid to `K` keypoints, then to a single vector.

### 2. Spatial information handling

- ACT preserves explicit spatial positions with a 2D positional embedding.
- Diffusion converts spatial activation patterns into expected keypoint coordinates via `SpatialSoftmax`.

### 3. Fusion with non-visual inputs

- ACT inserts latent/state/env/image features into one transformer token stream.
- Diffusion concatenates everything into one flat conditioning vector.

### 4. Policy core after visual encoding

- ACT feeds visual tokens into a transformer encoder-decoder for chunk prediction.
- Diffusion feeds a compact global condition into a 1D conditional U-Net over action trajectories.

### 5. Camera handling

- ACT uses one shared visual backbone and processes each camera independently, adding each camera's spatial tokens to the encoder sequence.
- Diffusion defaults to one shared RGB encoder across cameras, then concatenates the resulting per-camera vectors. It also supports separate encoders per camera.

References:

- ACT camera loop: `src/lerobot/policies/act/modeling_act.py:549-565`
- Diffusion shared/separate camera logic: `src/lerobot/policies/diffusion/modeling_diffusion.py:171-179`
- Diffusion camera fusion: `src/lerobot/policies/diffusion/modeling_diffusion.py:243-268`

### 6. ResNet normalization choices

- ACT uses `FrozenBatchNorm2d` in the ResNet path.
- Diffusion can replace BatchNorm with GroupNorm, and its default config enables that path when not using pretrained backbone weights.

References:

- ACT ResNet norm layer: `src/lerobot/policies/act/modeling_act.py:402-406`
- Diffusion BN -> GN replacement: `src/lerobot/policies/diffusion/modeling_diffusion.py:467-476`

## Default-Config Contrast

Under default configs in this repository:

- ACT defaults to `resnet18` with ImageNet pretrained weights:
  `src/lerobot/policies/act/configuration_act.py:106-107`
- Diffusion defaults to `resnet18` with `pretrained_backbone_weights=None`:
  `src/lerobot/policies/diffusion/configuration_diffusion.py:116-120`

This reinforces the intended usage patterns:

- ACT defaults are closer to a DETR-style visual token pipeline.
- Diffusion defaults are closer to a compact observation encoder feeding a conditional sequence model.

## Bottom Line

If you only remember one distinction, it should be this:

- ACT uses the ResNet as a spatial feature extractor whose output remains spatial and tokenized.
- Diffusion uses the ResNet as a perceptual frontend whose output is compressed into a global conditioning vector.
