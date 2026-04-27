#!/usr/bin/env bash
# Pre-download all model checkpoints for M5 multi-encoder sweep.
# Source this file or run it directly.
#
# Required env vars (with defaults):
#   MOCOV3_VITS_URL, VC1_VITB_URL, VOLTRON_CACHE_DIR, CPMAE_R301_CKPT

set -euo pipefail

MOCOV3_VITS_URL="${MOCOV3_VITS_URL:-https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar}"
VC1_VITB_URL="${VC1_VITB_URL:-https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth}"
VOLTRON_CACHE_DIR="${VOLTRON_CACHE_DIR:-$HOME/.voltron}"
CPMAE_R301_CKPT="${CPMAE_R301_CKPT:-results/M3_cpmae_improved/R301_cpmae/encoder_best.pt}"
CPMAE_HF_REPO="${CPMAE_HF_REPO:-haitong-ma/cpmae-pretrain}"

echo "Pre-downloading model checkpoints..."
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"

download_and_validate() {
  local url="$1"
  local fname
  fname=$(basename "${url%%\?*}")
  local fpath="${CACHE_DIR}/${fname}"

  if [[ -f "$fpath" ]]; then
    if python3 -c "import torch, sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$fpath" >/dev/null 2>&1; then
      echo "  ${fname} — cached"
      return 0
    else
      echo "  ${fname} — cached file is corrupted, re-downloading"
      rm -f "$fpath"
    fi
  fi

  echo "  Downloading ${fname} from ${url} ..."
  if ! curl -L -f -sS --retry 3 --retry-delay 2 --connect-timeout 30 -o "${fpath}.part" "$url"; then
    rm -f "${fpath}.part"
    echo "ERROR: curl failed to download ${url}" >&2
    return 1
  fi

  local size_bytes
  size_bytes=$(stat -c%s "${fpath}.part" 2>/dev/null || stat -f%z "${fpath}.part")
  if [[ "$size_bytes" -lt 1048576 ]]; then
    rm -f "${fpath}.part"
    echo "ERROR: downloaded file is only ${size_bytes} bytes." >&2
    return 1
  fi

  if ! python3 -c "import torch, sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "${fpath}.part" >/dev/null 2>&1; then
    mv "${fpath}.part" "${fpath}.corrupt"
    echo "ERROR: downloaded file is not a valid checkpoint." >&2
    return 1
  fi

  mv "${fpath}.part" "$fpath"
  echo "  ${fname} — done (${size_bytes} bytes)"
}

for url in "$MOCOV3_VITS_URL" "$VC1_VITB_URL"; do
  download_and_validate "$url"
done

# ── CP-MAE checkpoint ──────────────────────────────────────────────
if [[ -f "$CPMAE_R301_CKPT" ]]; then
  echo "  CP-MAE checkpoint — cached at ${CPMAE_R301_CKPT}"
else
  echo "  CP-MAE checkpoint not found at ${CPMAE_R301_CKPT}, downloading from ${CPMAE_HF_REPO}..."
  mkdir -p "$(dirname "$CPMAE_R301_CKPT")"
  HF_ENDPOINT=https://huggingface.co python3 - "$CPMAE_HF_REPO" "$CPMAE_R301_CKPT" <<'PYEOF' || { echo "  ERROR: failed to download CP-MAE checkpoint from ${CPMAE_HF_REPO}" >&2; return 1 2>/dev/null || exit 1; }
import os, sys, shutil
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
from huggingface_hub import hf_hub_download
repo_id, local_path = sys.argv[1], sys.argv[2]
# e.g. local_path = "results/M3_cpmae_improved/R301_cpmae/encoder_best.pt"
#   -> repo filename = "R301_cpmae/encoder_best.pt"
parts = local_path.replace("\\", "/").split("/")
subdir = parts[-2]  # R301_cpmae
basename = parts[-1]  # encoder_best.pt
repo_filename = f"{subdir}/{basename}"
downloaded = hf_hub_download(repo_id=repo_id, filename=repo_filename)
os.makedirs(os.path.dirname(local_path), exist_ok=True)
shutil.copy2(downloaded, local_path)
print(f"  CP-MAE checkpoint — downloaded to {local_path}")
PYEOF
fi

echo "Pre-downloading HuggingFace models (DINOv2, SigLIP)..."
python3 -c "
from transformers import Dinov2Model, SiglipVisionModel
for name in ['facebook/dinov2-small']:
    print(f'  Loading {name}...')
    Dinov2Model.from_pretrained(name)
    print(f'  {name} — cached')
print(f'  Loading google/siglip-base-patch16-224...')
SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224')
print(f'  google/siglip-base-patch16-224 — cached')
"

echo "Pre-downloading Voltron V-Cond checkpoint..."
mkdir -p "$VOLTRON_CACHE_DIR"
HF_ENDPOINT=https://huggingface.co python3 - <<PYEOF || { echo "  ERROR: failed to pre-download Voltron. Install: pip install voltron-robotics" >&2; return 1 2>/dev/null || exit 1; }
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
import voltron
model, _ = voltron.load('v-cond', freeze=True, cache='${VOLTRON_CACHE_DIR}')
print(f'  Voltron v-cond — cached at ${VOLTRON_CACHE_DIR}/v-cond/')
PYEOF

echo "Pre-download complete."
