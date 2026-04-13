#!/usr/bin/env python3
"""Training script wrapper that applies diagnostic image degradation.

Monkey-patches the make_dataset symbol INSIDE lerobot_train module (not just
in the factory module) so the degraded dataset is actually used during training.

Usage:
    python scripts/cpmae/train_with_degradation.py \
        --degrade_name=blur_sigma_5 \
        --contact_labels_dir=results/contact_labels \
        -- \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --policy.type=act \
        ...  (all standard lerobot-train args after --)

The args before -- are degradation-specific; args after -- are passed to lerobot-train.
"""

import sys
import argparse
from pathlib import Path

# Parse our custom args before --
custom_args = []
lerobot_args = []
found_separator = False
for arg in sys.argv[1:]:
    if arg == "--":
        found_separator = True
        continue
    if found_separator:
        lerobot_args.append(arg)
    else:
        custom_args.append(arg)

parser = argparse.ArgumentParser()
parser.add_argument("--degrade_name", type=str, required=True,
                    help="Degradation name (e.g., blur_sigma_5, degrade_contact_blur5)")
parser.add_argument("--contact_labels_dir", type=str, default="results/contact_labels",
                    help="Directory with contact label JSONs (for phase-aware degradation)")
custom_parsed = parser.parse_args(custom_args)

# Now set up sys.argv for lerobot's parser
sys.argv = ["lerobot-train"] + lerobot_args

# Add project root to path so we can import from scripts.cpmae
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.cpmae.diagnostic_dataset import DegradedDataset

# Import lerobot modules
import lerobot.datasets.factory as factory_module
import lerobot.scripts.lerobot_train as train_module
from lerobot.scripts.lerobot_train import train as lerobot_train

# Save originals
_original_factory_make_dataset = factory_module.make_dataset
_original_train_make_dataset = train_module.make_dataset


def _patched_make_dataset(cfg_arg, **kwargs):
    """Create dataset then wrap with degradation."""
    ds = _original_factory_make_dataset(cfg_arg, **kwargs)
    print(f"\n[DIAGNOSTIC] Wrapping dataset with degradation: {custom_parsed.degrade_name}")
    print(f"[DIAGNOSTIC] Image keys: {list(ds.meta.camera_keys)}")
    wrapped = DegradedDataset(
        dataset=ds,
        degrade_name=custom_parsed.degrade_name,
        contact_labels_dir=custom_parsed.contact_labels_dir,
    )
    print(f"[DIAGNOSTIC] Phase-aware: {wrapped.is_phase_aware}")
    if wrapped.is_phase_aware:
        print(f"[DIAGNOSTIC] Target phase: {wrapped.target_phase}")
    return wrapped


# Patch BOTH the factory module AND the train module's imported symbol
factory_module.make_dataset = _patched_make_dataset
train_module.make_dataset = _patched_make_dataset


def main():
    import draccus
    from lerobot.configs.train import TrainPipelineConfig

    cfg = draccus.parse(TrainPipelineConfig)
    cfg.validate()

    try:
        lerobot_train(cfg)
    finally:
        # Restore originals
        factory_module.make_dataset = _original_factory_make_dataset
        train_module.make_dataset = _original_train_make_dataset


if __name__ == "__main__":
    main()
