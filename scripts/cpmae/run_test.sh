#!/usr/bin/env bash
# One cell from run_hp_sweep.sh: ACT, bs=16, lr=1e-4, task_index=0 — quick smoke test
set -euo pipefail

export PYTHONWARNINGS="ignore"
export PYOPENGL_PLATFORM=egl

GPU=${1:-0}
REPO_ID="HuggingFaceVLA/libero"
TASK_IDX=0
BS=16
LR=1e-4
SEED=42

# Short run (sweep uses STEPS=100000)
STEPS=500
EVAL_FREQ=250
SAVE_FREQ=500
N_EVAL_EPISODES=2
EVAL_BATCH=2

RESULTS_DIR="results/M0_deep_smoke"
# Unique per invocation: do not mkdir the run dir here — lerobot refuses an existing output_dir when resume is false.
RUN_NAME="R000_act_bs${BS}_lr${LR}_task${TASK_IDX}_smoke"
RUN_DIR="$RESULTS_DIR/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

EPISODES=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('$REPO_ID')
task_name = meta.tasks[meta.tasks['task_index'] == $TASK_IDX].index[0]
eps = [ep['episode_index'] for ep in meta.episodes if task_name in ep['tasks']]
print('[' + ','.join(str(e) for e in sorted(eps)) + ']')
")

read -r SUITE ENV_TASK_ID <<< "$(python3 -c "
import sys, io, os
os.environ['LIBERO_QUIET'] = '1'
_real = sys.stdout; sys.stdout = io.StringIO()
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark
meta = LeRobotDatasetMetadata('$REPO_ID')
ds_tasks = {int(row['task_index']): name for name, row in meta.tasks.iterrows()}
task_name = ds_tasks[$TASK_IDX].strip().lower()
for suite_name in ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i in range(len(suite.tasks)):
        if suite.get_task(i).language.strip().lower() == task_name:
            sys.stdout = _real; print(suite_name, i); exit()
sys.stdout = _real; print('NOT_FOUND -1')
")"

echo "episodes=$EPISODES suite=$SUITE env_task_id=$ENV_TASK_ID"

MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --policy.type=act \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.freeze_backbone=true \
  --env.type=libero \
  --env.task=$SUITE \
  --env.task_ids="[$ENV_TASK_ID]" \
  --batch_size=$BS \
  --steps=$STEPS \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --eval.n_episodes=$N_EVAL_EPISODES \
  --eval.batch_size=$EVAL_BATCH \
  --seed=$SEED \
  --policy.optimizer_lr=$LR \
  --policy.optimizer_lr_backbone=$LR \
  --output_dir="$RUN_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false \
  --policy.push_to_hub=false