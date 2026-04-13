#!/usr/bin/env python3
"""CP-MAE (Contact-Phase Masked Autoencoder) pretraining on LIBERO images.

Two modes:
  - CP-MAE: Higher mask ratio for contact frames, weighted reconstruction loss
  - Uniform MAE: Standard MAE with uniform masking (control baseline)

Architecture: ViT-S/16 (384-dim, 6 heads, 12 blocks)
Data: Images extracted from HuggingFaceVLA/libero dataset

Usage:
    # CP-MAE pretraining
    python scripts/cpmae/pretrain_mae.py \
        --mode=cpmae \
        --output_dir=results/M3/R200_cpmae \
        --epochs=400 \
        --contact_labels_dir=results/contact_labels

    # Uniform MAE (control)
    python scripts/cpmae/pretrain_mae.py \
        --mode=uniform \
        --output_dir=results/M3/R201_uniform_mae \
        --epochs=400
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ViT components
class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for MAE."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        n_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, H, W) images
            mask: (B, N) binary mask. 1 = keep, 0 = mask out.
                  If None, all patches are kept.

        Returns:
            encoded: (B, N_visible + 1, D) encoded tokens (CLS + visible patches)
            ids_restore: (B, N) indices for restoring original order
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]

        # Add positional embedding (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]

        if mask is not None:
            # Keep only visible patches
            ids_keep = mask.nonzero(as_tuple=False)
            # Gather visible patches per sample
            # We need consistent length, so use argsort-based approach
            noise = torch.rand(B, N, device=x.device)
            noise[mask == 0] = 2.0  # Push masked to end
            ids_shuffle = noise.argsort(dim=1)
            ids_restore = ids_shuffle.argsort(dim=1)

            n_visible = int(mask.sum(dim=1).min().item())
            ids_keep = ids_shuffle[:, :n_visible]

            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        else:
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

        # Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore


class MAEDecoder(nn.Module):
    """Lightweight MAE decoder."""

    def __init__(
        self,
        n_patches,
        encoder_dim=384,
        decoder_dim=192,
        decoder_depth=4,
        decoder_heads=3,
        patch_size=16,
        in_chans=3,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_heads, mlp_ratio=4.0)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * in_chans)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x, ids_restore):
        """
        Args:
            x: (B, N_visible + 1, encoder_dim) from encoder
            ids_restore: (B, N) indices for restoring order
        """
        B, _, _ = x.shape
        N = ids_restore.shape[1]

        x = self.decoder_embed(x)  # (B, N_vis+1, decoder_dim)

        # Separate CLS and patch tokens
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # Append mask tokens
        n_masked = N - patch_tokens.shape[1]
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        full_tokens = torch.cat([patch_tokens, mask_tokens], dim=1)

        # Restore original order
        full_tokens = torch.gather(
            full_tokens, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        )

        # Add CLS back and positional embedding
        x = torch.cat([cls_token, full_tokens], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixel values for each patch
        x = self.decoder_pred(x[:, 1:, :])  # (B, N, patch_size^2 * 3)
        return x


class MAE(nn.Module):
    """Masked Autoencoder with ViT backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        encoder_dim=384,
        encoder_depth=12,
        encoder_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        decoder_heads=3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        n_patches = (img_size // patch_size) ** 2

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
        )
        self.decoder = MAEDecoder(
            n_patches=n_patches,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            patch_size=patch_size,
            in_chans=in_chans,
        )

    def patchify(self, imgs):
        """Convert images to patch targets."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def forward(self, imgs, mask):
        """
        Args:
            imgs: (B, C, H, W)
            mask: (B, N) binary mask. 1 = visible, 0 = masked.

        Returns:
            loss: reconstruction MSE on masked patches
            pred: (B, N, patch_size^2 * 3) predictions
        """
        encoded, ids_restore = self.encoder(imgs, mask)
        pred = self.decoder(encoded, ids_restore)
        target = self.patchify(imgs)

        # Loss on masked patches only
        loss_mask = 1.0 - mask.float()  # 1 where masked
        loss = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return loss, pred


# Dataset for MAE pretraining
class LiberoImageDataset(Dataset):
    """Extracts images from HuggingFaceVLA/libero for MAE pretraining."""

    def __init__(
        self,
        repo_id="HuggingFaceVLA/libero",
        episodes=None,
        image_key="observation.images.image",
        img_size=224,
        contact_labels_dir=None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(repo_id=repo_id, episodes=episodes)
        self.image_key = image_key
        self.img_size = img_size

        # Load contact labels if available
        self.contact_labels = {}
        if contact_labels_dir is not None:
            contact_dir = Path(contact_labels_dir)
            for label_file in sorted(contact_dir.glob("task_*_contacts.json")):
                with open(label_file) as f:
                    data = json.load(f)
                for ep_str, ep_data in data["episodes"].items():
                    self.contact_labels[int(ep_str)] = ep_data["labels"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_key]  # (C, H, W), already float [0, 1]

        # Resize if needed
        if img.shape[-1] != self.img_size or img.shape[-2] != self.img_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        ep_idx = item["episode_index"].item() if torch.is_tensor(item["episode_index"]) else item["episode_index"]
        frame_idx = item["frame_index"].item() if torch.is_tensor(item["frame_index"]) else item["frame_index"]

        # Determine contact label
        is_contact = 0
        if ep_idx in self.contact_labels:
            labels = self.contact_labels[ep_idx]
            if frame_idx < len(labels):
                is_contact = labels[frame_idx]

        return {
            "image": img,
            "is_contact": is_contact,
            "episode_index": ep_idx,
            "frame_index": frame_idx,
        }


def generate_mask(batch_size, n_patches, mask_ratio, device):
    """Generate random mask. Returns (B, N) with 1=visible, 0=masked."""
    n_visible = int(n_patches * (1 - mask_ratio))
    noise = torch.rand(batch_size, n_patches, device=device)
    ids_shuffle = noise.argsort(dim=1)
    mask = torch.zeros(batch_size, n_patches, device=device)
    mask.scatter_(1, ids_shuffle[:, :n_visible], 1.0)
    return mask


def generate_cpmae_mask(is_contact, n_patches, contact_mask_ratio, transit_mask_ratio, device):
    """Generate contact-phase-aware masks via split-batch approach.

    Contact frames get higher mask ratio (harder reconstruction).
    Transit frames get standard mask ratio.

    Each sub-group gets a uniform number of visible patches so the encoder
    forward pass works correctly (no min-across-batch truncation).
    Returns masks and the split indices for separate forward passes.
    """
    B = is_contact.shape[0]
    contact_idx = (is_contact == 1).nonzero(as_tuple=True)[0]
    transit_idx = (is_contact == 0).nonzero(as_tuple=True)[0]

    mask = torch.zeros(B, n_patches, device=device)

    # Generate masks for contact samples
    if len(contact_idx) > 0:
        n_c = len(contact_idx)
        n_visible_c = int(n_patches * (1 - contact_mask_ratio))
        noise_c = torch.rand(n_c, n_patches, device=device)
        ids_c = noise_c.argsort(dim=1)
        for j, i in enumerate(contact_idx):
            mask[i].scatter_(0, ids_c[j, :n_visible_c], 1.0)

    # Generate masks for transit samples
    if len(transit_idx) > 0:
        n_t = len(transit_idx)
        n_visible_t = int(n_patches * (1 - transit_mask_ratio))
        noise_t = torch.rand(n_t, n_patches, device=device)
        ids_t = noise_t.argsort(dim=1)
        for j, i in enumerate(transit_idx):
            mask[i].scatter_(0, ids_t[j, :n_visible_t], 1.0)

    return mask, contact_idx, transit_idx


def train_mae(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    # Dataset
    contact_labels_dir = args.contact_labels_dir if args.mode == "cpmae" else None
    dataset = LiberoImageDataset(
        repo_id=args.repo_id,
        image_key=args.image_key,
        img_size=args.img_size,
        contact_labels_dir=contact_labels_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset size: {len(dataset)} frames")
    print(f"Batches per epoch: {len(dataloader)}")

    # Model
    n_patches = (args.img_size // args.patch_size) ** 2
    model = MAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Total params: {n_params / 1e6:.1f}M (encoder: {n_encoder_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine annealing with warmup
    warmup_steps = args.warmup_epochs * len(dataloader)
    total_steps = args.epochs * len(dataloader)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    log_history = []
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_contact_loss = 0.0
        epoch_transit_loss = 0.0
        n_contact = 0
        n_transit = 0
        t0 = time.time()

        for batch in dataloader:
            imgs = batch["image"].to(device)
            is_contact = batch["is_contact"].to(device)
            B = imgs.shape[0]

            # Generate masks and compute loss
            if args.mode == "cpmae":
                mask, contact_idx, transit_idx = generate_cpmae_mask(
                    is_contact, n_patches,
                    contact_mask_ratio=args.contact_mask_ratio,
                    transit_mask_ratio=args.transit_mask_ratio,
                    device=device,
                )

                # Split-batch forward: run contact and transit sub-batches separately
                # so each group has consistent visible-patch count for the encoder.
                per_sample_loss = torch.zeros(B, device=device)

                if len(contact_idx) > 0:
                    c_imgs = imgs[contact_idx]
                    c_mask = mask[contact_idx]
                    c_loss_raw, c_pred = model(c_imgs, c_mask)
                    # Compute per-sample loss for contact
                    c_target = model.patchify(c_imgs)
                    c_loss_mask = 1.0 - c_mask
                    c_per_sample = ((c_pred - c_target) ** 2).mean(dim=-1)
                    c_per_sample = (c_per_sample * c_loss_mask).sum(dim=1) / c_loss_mask.sum(dim=1).clamp(min=1)
                    per_sample_loss[contact_idx] = c_per_sample

                if len(transit_idx) > 0:
                    t_imgs = imgs[transit_idx]
                    t_mask = mask[transit_idx]
                    t_loss_raw, t_pred = model(t_imgs, t_mask)
                    t_target = model.patchify(t_imgs)
                    t_loss_mask = 1.0 - t_mask
                    t_per_sample = ((t_pred - t_target) ** 2).mean(dim=-1)
                    t_per_sample = (t_per_sample * t_loss_mask).sum(dim=1) / t_loss_mask.sum(dim=1).clamp(min=1)
                    per_sample_loss[transit_idx] = t_per_sample

                # Apply contact-weighted loss
                weights = torch.where(
                    is_contact.bool(),
                    torch.tensor(args.contact_loss_weight, device=device),
                    torch.tensor(1.0, device=device),
                )
                loss = (per_sample_loss * weights).sum() / weights.sum()

                # Track per-phase losses
                if len(contact_idx) > 0:
                    epoch_contact_loss += per_sample_loss[contact_idx].sum().item()
                    n_contact += len(contact_idx)
                if len(transit_idx) > 0:
                    epoch_transit_loss += per_sample_loss[transit_idx].sum().item()
                    n_transit += len(transit_idx)
            else:
                mask = generate_mask(B, n_patches, args.uniform_mask_ratio, device)
                loss, pred = model(imgs, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * B
            global_step += 1

        epoch_loss /= len(dataset)
        dt = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "loss": epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": dt,
        }

        if args.mode == "cpmae":
            if n_contact > 0:
                log_entry["contact_loss"] = epoch_contact_loss / n_contact
            if n_transit > 0:
                log_entry["transit_loss"] = epoch_transit_loss / n_transit
            log_entry["contact_ratio"] = n_contact / max(1, n_contact + n_transit)

        log_history.append(log_entry)

        if epoch % args.log_freq == 0:
            msg = f"Epoch {epoch}/{args.epochs} | loss={epoch_loss:.6f} | lr={log_entry['lr']:.2e} | {dt:.1f}s"
            if "contact_loss" in log_entry:
                msg += f" | c_loss={log_entry['contact_loss']:.6f} | t_loss={log_entry['transit_loss']:.6f}"
            print(msg)

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.encoder.state_dict(), output_dir / "encoder_best.pt")

        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Save final
    torch.save(model.encoder.state_dict(), output_dir / "encoder_final.pt")
    torch.save(model.state_dict(), output_dir / "model_final.pt")

    # Save config and log
    config = vars(args)
    config["n_params"] = n_params
    config["n_encoder_params"] = n_encoder_params
    config["best_loss"] = best_loss
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Encoder saved to: {output_dir / 'encoder_final.pt'}")


def main():
    parser = argparse.ArgumentParser(description="CP-MAE / Uniform MAE Pretraining")

    # Mode
    parser.add_argument("--mode", type=str, required=True, choices=["cpmae", "uniform"],
                        help="Pretraining mode: cpmae or uniform")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Data
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--image_key", type=str, default="observation.images.image")
    parser.add_argument("--contact_labels_dir", type=str, default="results/contact_labels")

    # Architecture
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--encoder_dim", type=int, default=384)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=6)
    parser.add_argument("--decoder_dim", type=int, default=192)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=3)

    # Masking
    parser.add_argument("--uniform_mask_ratio", type=float, default=0.75)
    parser.add_argument("--contact_mask_ratio", type=float, default=0.90)
    parser.add_argument("--transit_mask_ratio", type=float, default=0.75)
    parser.add_argument("--contact_loss_weight", type=float, default=2.0)

    # Training
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    # Logging
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100)

    args = parser.parse_args()
    train_mae(args)


if __name__ == "__main__":
    main()
