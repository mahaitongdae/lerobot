#!/usr/bin/env python3
"""Legacy helper that forwards CP-MAE convenience flags into ``lerobot-train``.

ACT now supports ``vision_backbone=cpmae`` natively, so this script keeps older
launch commands working by translating ``--cpmae_*`` flags into standard policy
config overrides before delegating to the normal training pipeline.

Usage:
    python scripts/cpmae/train_with_cpmae.py \
        --cpmae_checkpoint=results/M3/R200_cpmae/encoder_final.pt \
        --cpmae_freeze=true \
        -- \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --policy.type=act \
        --policy.vision_backbone=cpmae \
        ...  (all standard lerobot-train args after --)
"""

import sys

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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cpmae_checkpoint", type=str, required=True,
                    help="Path to CP-MAE encoder checkpoint (.pt)")
parser.add_argument("--cpmae_freeze", type=str, default="true",
                    help="Whether to freeze the CP-MAE encoder (true/false)")
custom_parsed = parser.parse_args(custom_args)

freeze = custom_parsed.cpmae_freeze.lower() in ("true", "1", "yes")

# Keep legacy callers working while routing through the native ACT cpmae path.
translated_args = [
    f"--policy.cpmae_checkpoint_path={custom_parsed.cpmae_checkpoint}",
    f"--policy.freeze_backbone={'true' if freeze else 'false'}",
]
if not any(arg.startswith("--policy.backbone_input_norm=") for arg in lerobot_args):
    translated_args.append("--policy.backbone_input_norm=identity")
if not any(arg == "--policy.vision_backbone=cpmae" or arg.startswith("--policy.vision_backbone=") for arg in lerobot_args):
    translated_args.append("--policy.vision_backbone=cpmae")

# Now set up sys.argv for lerobot's parser and run training
sys.argv = ["lerobot-train"] + translated_args + lerobot_args

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train as lerobot_train

cfg = draccus.parse(TrainPipelineConfig)
cfg.validate()
lerobot_train(cfg)
