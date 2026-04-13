#!/usr/bin/env python3
"""Training script that loads a CP-MAE encoder into ACT before running lerobot-train.

This applies the monkey-patch from cpmae_backbone.py, then delegates to lerobot's
standard training pipeline.

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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cpmae_checkpoint", type=str, required=True,
                    help="Path to CP-MAE encoder checkpoint (.pt)")
parser.add_argument("--cpmae_freeze", type=str, default="true",
                    help="Whether to freeze the CP-MAE encoder (true/false)")
custom_parsed = parser.parse_args(custom_args)

freeze = custom_parsed.cpmae_freeze.lower() in ("true", "1", "yes")

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Apply the CP-MAE monkey-patch before lerobot imports ACT
from scripts.cpmae.cpmae_backbone import patch_act_for_cpmae

patch_act_for_cpmae(
    checkpoint_path=custom_parsed.cpmae_checkpoint,
    freeze=freeze,
)

# Now set up sys.argv for lerobot's parser and run training
sys.argv = ["lerobot-train"] + lerobot_args

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train as lerobot_train

cfg = draccus.parse(TrainPipelineConfig)
cfg.validate()
lerobot_train(cfg)
