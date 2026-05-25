Add last to checkpoints

```bash
cd /mnt/shared/haitongma/wdir/lerobot/results/ssl_sweep_allsuites
for run in SSL_dp_libero_spatial_byol SSL_dp_libero_spatial_imagenet SSL_dp_libero_spatial_moco_v2 SSL_dp_libero_spatial_simclr; do
  latest=$(ls "$run/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
  ln -sfn "$latest" "$run/checkpoints/last"
done
```

```bash
cd /app/results/ssl_sweep_allsuites
for run in SSL_dp_libero_spatial_byol SSL_dp_libero_spatial_imagenet SSL_dp_libero_spatial_moco_v2 SSL_dp_libero_spatial_simclr; do
  latest=$(ls "$run/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
  ln -sfn "$latest" "$run/checkpoints/last"
done
```

CUDA_VISIBLE_DEVICES=1 lerobot-train --dataset.repo_id=HuggingFaceVLA/libero --policy.type=act --policy.vision_backbone=cpmae --policy.cpmae_checkpoint_path=results/M3_cpmae/R200_cpmae/encoder_final.pt --policy.freeze_backbone=true --policy.num_tasks=10 --policy.task_embed_dim=64 --policy.task_index_offset=10 --env.type=libero --env.task=libero_goal --env.task_ids=\[8\,9\,3\,6\,2\,5\,7\,1\,4\,0\] --batch_size=64 --steps=100 --eval_freq=50 --save_freq=25000 --eval.n_episodes=20 --eval.batch_size=20 --seed=42 --policy.optimizer_lr=5e-5 --policy.optimizer_lr_backbone=5e-5 --output_dir=results/cpmae_sweep_allsuites/M3b_act_libero_goal_cpmae_frozen --job_name=M3b_act_libero_goal_cpmae_frozen --wandb.enable=false --policy.push_to_hub=false