#!/usr/bin/env python3
"""M4d: Representation analysis — CKA similarity, attention maps, reconstruction MSE.

Compares CP-MAE vs Uniform MAE ViT encoders.

Analyses:
  1. CKA similarity between CP-MAE and Uniform MAE (layer-by-layer)
  2. Attention entropy on contact vs transit frames (higher entropy = more diffuse)
  3. Reconstruction MSE breakdown (contact vs transit)

Usage:
    python scripts/cpmae/analyze_representations.py \
        --cpmae_checkpoint=results/M3_cpmae/R200_cpmae/encoder_final.pt \
        --umae_checkpoint=results/M3_cpmae/R201_uniform_mae/encoder_final.pt \
        --contact_labels_dir=results/contact_labels \
        --output_dir=results/M4_ablations/analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import our MAE components
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.cpmae.pretrain_mae import (
    MAE, ViTEncoder, LiberoImageDataset, generate_mask
)


def load_encoder(checkpoint_path: str, device: torch.device) -> ViTEncoder:
    """Load a pretrained ViT encoder from checkpoint."""
    encoder = ViTEncoder(
        img_size=224, patch_size=16, embed_dim=384,
        depth=12, n_heads=6,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def load_imagenet_resnet18(device: torch.device):
    """Load ImageNet-pretrained ResNet18 for comparison."""
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    return model


# =========================================================================
# CKA (Centered Kernel Alignment)
# =========================================================================

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (N, D1) representations from model 1
        Y: (N, D2) representations from model 2

    Returns:
        CKA similarity in [0, 1]
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()

    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return cka.item()


def extract_vit_layer_features(encoder: ViTEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
    """Extract intermediate features from each transformer block.

    Returns list of (B, N+1, D) tensors, one per block.
    """
    features = []
    with torch.no_grad():
        x = encoder.patch_embed(imgs)
        x = x + encoder.pos_embed[:, 1:, :]
        cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(imgs.shape[0], -1, -1), x], dim=1)

        for blk in encoder.blocks:
            x = blk(x)
            features.append(x.clone())

        x = encoder.norm(x)
        features.append(x)  # Final normed output

    return features


def extract_resnet_layer_features(model, imgs: torch.Tensor) -> list[torch.Tensor]:
    """Extract intermediate features from ResNet layers."""
    features = []
    with torch.no_grad():
        x = model.conv1(imgs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            x = layer(x)
            # Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
            features.append(x.flatten(start_dim=1))

        # Global average pool
        x = model.avgpool(x)
        features.append(x.flatten(start_dim=1))

    return features


def compute_cka_matrix(
    encoder_a: ViTEncoder,
    encoder_b: ViTEncoder,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 1000,
) -> np.ndarray:
    """Compute layer-wise CKA between two ViT encoders."""
    all_feats_a = None
    all_feats_b = None
    n_collected = 0

    for batch in dataloader:
        imgs = batch["image"].to(device)
        feats_a = extract_vit_layer_features(encoder_a, imgs)
        feats_b = extract_vit_layer_features(encoder_b, imgs)

        if all_feats_a is None:
            n_layers_a = len(feats_a)
            n_layers_b = len(feats_b)
            all_feats_a = [[] for _ in range(n_layers_a)]
            all_feats_b = [[] for _ in range(n_layers_b)]

        for i, f in enumerate(feats_a):
            # Use CLS token representation
            all_feats_a[i].append(f[:, 0, :].cpu())
        for i, f in enumerate(feats_b):
            all_feats_b[i].append(f[:, 0, :].cpu())

        n_collected += imgs.shape[0]
        if n_collected >= n_samples:
            break

    # Concatenate
    for i in range(len(all_feats_a)):
        all_feats_a[i] = torch.cat(all_feats_a[i])[:n_samples]
    for i in range(len(all_feats_b)):
        all_feats_b[i] = torch.cat(all_feats_b[i])[:n_samples]

    # Compute CKA matrix
    cka_matrix = np.zeros((len(all_feats_a), len(all_feats_b)))
    for i in range(len(all_feats_a)):
        for j in range(len(all_feats_b)):
            cka_matrix[i, j] = linear_cka(all_feats_a[i], all_feats_b[j])
            print(f"  CKA[layer {i}, layer {j}] = {cka_matrix[i, j]:.4f}")

    return cka_matrix


# =========================================================================
# Attention Map Analysis
# =========================================================================

def extract_attention_maps(encoder: ViTEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
    """Extract attention maps from each transformer block.

    Returns list of (B, n_heads, N+1, N+1) attention weight tensors.
    """
    attn_maps = []

    with torch.no_grad():
        x = encoder.patch_embed(imgs)
        x = x + encoder.pos_embed[:, 1:, :]
        cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(imgs.shape[0], -1, -1), x], dim=1)

        for blk in encoder.blocks:
            # Manually compute attention to capture weights
            B, N, C = x.shape
            norm_x = blk.norm1(x)
            qkv = blk.attn.qkv(norm_x).reshape(B, N, 3, blk.attn.n_heads, blk.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
            attn = attn.softmax(dim=-1)
            attn_maps.append(attn.cpu())

            # Continue forward pass
            x = blk(x)

    return attn_maps


def analyze_attention(
    encoder: ViTEncoder,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 200,
) -> dict:
    """Analyze attention patterns on contact vs transit frames."""
    contact_cls_attn = []  # CLS attention over patches for contact frames
    transit_cls_attn = []

    n_collected = 0
    for batch in dataloader:
        imgs = batch["image"].to(device)
        is_contact = batch["is_contact"]

        attn_maps = extract_attention_maps(encoder, imgs)

        # Use last layer attention, average across heads
        last_attn = attn_maps[-1].mean(dim=1)  # (B, N+1, N+1)
        # CLS token's attention to patches (row 0, cols 1:)
        cls_to_patches = last_attn[:, 0, 1:]  # (B, N_patches)
        # Renormalize to sum to 1 (since we dropped CLS self-attention mass)
        cls_to_patches = cls_to_patches / cls_to_patches.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        for i in range(imgs.shape[0]):
            if is_contact[i] == 1:
                contact_cls_attn.append(cls_to_patches[i].numpy())
            else:
                transit_cls_attn.append(cls_to_patches[i].numpy())

        n_collected += imgs.shape[0]
        if n_collected >= n_samples:
            break

    results = {}
    if contact_cls_attn:
        contact_attn = np.stack(contact_cls_attn)
        results["contact"] = {
            "mean_attn": contact_attn.mean(axis=0).tolist(),
            "entropy": float(-np.sum(contact_attn * np.log(contact_attn + 1e-10), axis=-1).mean()),
            "n_samples": len(contact_cls_attn),
        }
    if transit_cls_attn:
        transit_attn = np.stack(transit_cls_attn)
        results["transit"] = {
            "mean_attn": transit_attn.mean(axis=0).tolist(),
            "entropy": float(-np.sum(transit_attn * np.log(transit_attn + 1e-10), axis=-1).mean()),
            "n_samples": len(transit_cls_attn),
        }

    return results


# =========================================================================
# Reconstruction MSE Analysis
# =========================================================================

def compute_reconstruction_mse(
    model: MAE,
    dataloader: DataLoader,
    device: torch.device,
    mask_ratio: float = 0.75,
    n_samples: int = 500,
) -> dict:
    """Compute reconstruction MSE separately for contact vs transit frames."""
    model.eval()
    n_patches = (model.encoder.patch_embed.img_size // model.encoder.patch_embed.patch_size) ** 2

    contact_mse = []
    transit_mse = []
    all_mse = []

    n_collected = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            is_contact = batch["is_contact"]
            B = imgs.shape[0]

            mask = generate_mask(B, n_patches, mask_ratio, device)
            encoded, ids_restore = model.encoder(imgs, mask)
            pred = model.decoder(encoded, ids_restore)
            target = model.patchify(imgs)

            loss_mask = 1.0 - mask
            per_sample = ((pred - target) ** 2).mean(dim=-1)
            per_sample = (per_sample * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

            for i in range(B):
                mse_val = per_sample[i].item()
                all_mse.append(mse_val)
                if is_contact[i] == 1:
                    contact_mse.append(mse_val)
                else:
                    transit_mse.append(mse_val)

            n_collected += B
            if n_collected >= n_samples:
                break

    results = {
        "overall_mse": float(np.mean(all_mse)),
        "overall_std": float(np.std(all_mse)),
        "n_total": len(all_mse),
    }
    if contact_mse:
        results["contact_mse"] = float(np.mean(contact_mse))
        results["contact_std"] = float(np.std(contact_mse))
        results["n_contact"] = len(contact_mse)
    if transit_mse:
        results["transit_mse"] = float(np.mean(transit_mse))
        results["transit_std"] = float(np.std(transit_mse))
        results["n_transit"] = len(transit_mse)

    return results


# =========================================================================
# Patch-Occlusion Sensitivity (Appendix)
# =========================================================================

def compute_patch_occlusion_sensitivity(
    encoder: ViTEncoder,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 100,
    grid_size: int = 14,
) -> dict:
    """Measure how occluding each patch region changes the encoder output.

    For each image, mask one patch at a time (set to mean pixel value),
    encode, and measure the L2 change in the CLS representation.
    This gives a per-patch "importance" map without requiring a policy.

    Returns per-patch sensitivity maps averaged over contact/transit frames.
    """
    n_patches = grid_size * grid_size
    patch_size = 224 // grid_size  # assumes 224x224 input

    contact_sensitivity = []
    transit_sensitivity = []
    n_collected = 0

    for batch in dataloader:
        imgs = batch["image"].to(device)
        is_contact = batch["is_contact"]
        B = imgs.shape[0]

        with torch.no_grad():
            # Get baseline CLS representation (no mask)
            baseline_encoded, _ = encoder(imgs, mask=None)
            baseline_cls = baseline_encoded[:, 0, :]  # (B, D)

            # For each patch position, occlude and measure change
            sensitivity = torch.zeros(B, n_patches, device=device)

            for p in range(n_patches):
                row = p // grid_size
                col = p % grid_size
                y0, y1 = row * patch_size, (row + 1) * patch_size
                x0, x1 = col * patch_size, (col + 1) * patch_size

                occluded = imgs.clone()
                # Set patch to mean pixel value (approximately gray)
                occluded[:, :, y0:y1, x0:x1] = 0.5

                occluded_encoded, _ = encoder(occluded, mask=None)
                occluded_cls = occluded_encoded[:, 0, :]

                # L2 distance in CLS space
                sensitivity[:, p] = (baseline_cls - occluded_cls).norm(dim=-1)

        for i in range(B):
            sens = sensitivity[i].cpu().numpy()
            if is_contact[i] == 1:
                contact_sensitivity.append(sens)
            else:
                transit_sensitivity.append(sens)

        n_collected += B
        if n_collected >= n_samples:
            break

    results = {}
    if contact_sensitivity:
        contact_sens = np.stack(contact_sensitivity)
        results["contact"] = {
            "mean_sensitivity": contact_sens.mean(axis=0).tolist(),
            "mean_total": float(contact_sens.mean()),
            "n_samples": len(contact_sensitivity),
        }
    if transit_sensitivity:
        transit_sens = np.stack(transit_sensitivity)
        results["transit"] = {
            "mean_sensitivity": transit_sens.mean(axis=0).tolist(),
            "mean_total": float(transit_sens.mean()),
            "n_samples": len(transit_sensitivity),
        }

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="M4d: Representation analysis")
    parser.add_argument("--cpmae_checkpoint", type=str, required=True,
                        help="Path to CP-MAE encoder checkpoint")
    parser.add_argument("--umae_checkpoint", type=str, required=True,
                        help="Path to Uniform MAE encoder checkpoint")
    parser.add_argument("--cpmae_model_checkpoint", type=str, default=None,
                        help="Path to full CP-MAE model (encoder+decoder) for reconstruction analysis")
    parser.add_argument("--umae_model_checkpoint", type=str, default=None,
                        help="Path to full Uniform MAE model for reconstruction analysis")
    parser.add_argument("--contact_labels_dir", type=str, default="results/contact_labels")
    parser.add_argument("--output_dir", type=str, default="results/M4_ablations/analysis")
    parser.add_argument("--n_cka_samples", type=int, default=1000)
    parser.add_argument("--n_attn_samples", type=int, default=200)
    parser.add_argument("--n_recon_samples", type=int, default=500)
    parser.add_argument("--n_occlusion_samples", type=int, default=100)
    parser.add_argument("--run_occlusion", action="store_true",
                        help="Run patch-occlusion sensitivity analysis (appendix, slower)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = LiberoImageDataset(
        repo_id="HuggingFaceVLA/libero",
        contact_labels_dir=args.contact_labels_dir,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Dataset: {len(dataset)} frames")

    # Load encoders
    print("\nLoading encoders...")
    cpmae_enc = load_encoder(args.cpmae_checkpoint, device)
    umae_enc = load_encoder(args.umae_checkpoint, device)
    print("  CP-MAE and Uniform MAE encoders loaded")

    all_results = {}

    # ---- CKA Analysis ----
    print("\n" + "=" * 60)
    print("CKA Analysis: CP-MAE vs Uniform MAE")
    print("=" * 60)

    cka_cpmae_umae = compute_cka_matrix(cpmae_enc, umae_enc, dataloader, device, args.n_cka_samples)
    all_results["cka_cpmae_vs_umae"] = cka_cpmae_umae.tolist()

    print(f"\nDiagonal CKA (same-layer): {[f'{cka_cpmae_umae[i,i]:.3f}' for i in range(min(cka_cpmae_umae.shape))]}")

    # Self-CKA (same model, layer similarity)
    print("\nCKA Analysis: CP-MAE self-similarity")
    cka_cpmae_self = compute_cka_matrix(cpmae_enc, cpmae_enc, dataloader, device, args.n_cka_samples)
    all_results["cka_cpmae_self"] = cka_cpmae_self.tolist()

    # ---- Attention Analysis ----
    print("\n" + "=" * 60)
    print("Attention Map Analysis")
    print("=" * 60)

    print("\nCP-MAE attention analysis...")
    cpmae_attn = analyze_attention(cpmae_enc, dataloader, device, args.n_attn_samples)
    all_results["attention_cpmae"] = cpmae_attn

    if "contact" in cpmae_attn and "transit" in cpmae_attn:
        print(f"  CP-MAE attention entropy — contact: {cpmae_attn['contact']['entropy']:.4f}, "
              f"transit: {cpmae_attn['transit']['entropy']:.4f}")

    print("\nUniform MAE attention analysis...")
    umae_attn = analyze_attention(umae_enc, dataloader, device, args.n_attn_samples)
    all_results["attention_umae"] = umae_attn

    if "contact" in umae_attn and "transit" in umae_attn:
        print(f"  UMAE attention entropy — contact: {umae_attn['contact']['entropy']:.4f}, "
              f"transit: {umae_attn['transit']['entropy']:.4f}")

    # ---- Reconstruction MSE Analysis ----
    if args.cpmae_model_checkpoint and args.umae_model_checkpoint:
        print("\n" + "=" * 60)
        print("Reconstruction MSE Analysis")
        print("=" * 60)

        # Load full models (encoder + decoder)
        n_patches = (224 // 16) ** 2
        cpmae_model = MAE().to(device)
        cpmae_model.load_state_dict(
            torch.load(args.cpmae_model_checkpoint, map_location=device, weights_only=True)
        )
        cpmae_model.eval()

        umae_model = MAE().to(device)
        umae_model.load_state_dict(
            torch.load(args.umae_model_checkpoint, map_location=device, weights_only=True)
        )
        umae_model.eval()

        print("\nCP-MAE reconstruction...")
        cpmae_recon = compute_reconstruction_mse(cpmae_model, dataloader, device, 0.75, args.n_recon_samples)
        all_results["reconstruction_cpmae"] = cpmae_recon
        print(f"  Overall MSE: {cpmae_recon['overall_mse']:.6f}")
        if "contact_mse" in cpmae_recon:
            print(f"  Contact MSE: {cpmae_recon['contact_mse']:.6f}, Transit MSE: {cpmae_recon['transit_mse']:.6f}")

        print("\nUniform MAE reconstruction...")
        umae_recon = compute_reconstruction_mse(umae_model, dataloader, device, 0.75, args.n_recon_samples)
        all_results["reconstruction_umae"] = umae_recon
        print(f"  Overall MSE: {umae_recon['overall_mse']:.6f}")
        if "contact_mse" in umae_recon:
            print(f"  Contact MSE: {umae_recon['contact_mse']:.6f}, Transit MSE: {umae_recon['transit_mse']:.6f}")

        # Also test CP-MAE at its training mask ratio (0.90)
        print("\nCP-MAE reconstruction at 90% mask ratio...")
        cpmae_recon_90 = compute_reconstruction_mse(cpmae_model, dataloader, device, 0.90, args.n_recon_samples)
        all_results["reconstruction_cpmae_90mask"] = cpmae_recon_90
        print(f"  Overall MSE: {cpmae_recon_90['overall_mse']:.6f}")
    else:
        print("\nSkipping reconstruction analysis (provide --cpmae_model_checkpoint and --umae_model_checkpoint)")

    # ---- Patch-Occlusion Sensitivity (appendix) ----
    if args.run_occlusion:
        print("\n" + "=" * 60)
        print("Patch-Occlusion Sensitivity (Appendix)")
        print("=" * 60)

        print("\nCP-MAE patch occlusion...")
        cpmae_occ = compute_patch_occlusion_sensitivity(
            cpmae_enc, dataloader, device, args.n_occlusion_samples
        )
        all_results["occlusion_cpmae"] = cpmae_occ
        if "contact" in cpmae_occ and "transit" in cpmae_occ:
            print(f"  CP-MAE mean sensitivity — contact: {cpmae_occ['contact']['mean_total']:.4f}, "
                  f"transit: {cpmae_occ['transit']['mean_total']:.4f}")

        print("\nUniform MAE patch occlusion...")
        umae_occ = compute_patch_occlusion_sensitivity(
            umae_enc, dataloader, device, args.n_occlusion_samples
        )
        all_results["occlusion_umae"] = umae_occ
        if "contact" in umae_occ and "transit" in umae_occ:
            print(f"  UMAE mean sensitivity — contact: {umae_occ['contact']['mean_total']:.4f}, "
                  f"transit: {umae_occ['transit']['mean_total']:.4f}")
    else:
        print("\nSkipping patch-occlusion analysis (use --run_occlusion to enable)")

    # Save results
    results_file = output_dir / "analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"CKA diagonal (CP-MAE vs UMAE): "
          f"early={cka_cpmae_umae[0,0]:.3f}, mid={cka_cpmae_umae[6,6]:.3f}, "
          f"late={cka_cpmae_umae[-1,-1]:.3f}")

    if "attention_cpmae" in all_results and "contact" in all_results["attention_cpmae"]:
        c_ent = all_results["attention_cpmae"]["contact"]["entropy"]
        t_ent = all_results["attention_cpmae"]["transit"]["entropy"]
        print(f"CP-MAE attention entropy: contact={c_ent:.4f}, transit={t_ent:.4f}, delta={c_ent - t_ent:.4f}")


if __name__ == "__main__":
    main()
