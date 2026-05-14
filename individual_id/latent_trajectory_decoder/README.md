# Latent Trajectory Decoder

Small exploration for asking whether trajectories through UMAP or latent space carry individual-ID information.

The experiment is intentionally unsupervised until the probe:

1. Load a saved UMAP HDBSCAN points artifact with `xy`, or a latent feature cache with `features`.
2. Split recordings into train/validation sets within each bird.
3. Train a roughly 500k-parameter, 2-layer causal transformer to predict the next UMAP coordinate from previous coordinates.
4. Freeze the transformer and train a linear probe on pooled decoder states to predict bird ID.
5. Save a GIF comparing a held-out true coordinate trajectory against an autoregressive decoder rollout.

Bird labels are not used by the decoder loss. They are only used for the final probe and metrics.

For high-dimensional latent features, prediction is trained in the full feature space. The GIF is only a local 2D PCA projection of one held-out true/predicted trajectory.

Example:

```bash
python individual_id/latent_trajectory_decoder/train_probe.py \
  --umap_points_npz results/individual_id_umap/stats_affinity_w25_h20_recsvd20_alpha2_hdbscan_all8_largest_working/zf_w25_h20_stats_pca512_recsvd20_a2_hdbscan_nn50_cosine/songmae_pool_stats_sliding_w25_h20_recsvd20_a2_pca_whiten_l2512_hdbscan_points.npz \
  --out_dir results/individual_id_latent_trajectory_decoder/zf_recsvd20_w25_h20 \
  --epochs 20 \
  --probe_epochs 200 \
  --d_model 144 \
  --layers 2
```

Useful quick smoke test:

```bash
python individual_id/latent_trajectory_decoder/train_probe.py \
  --umap_points_npz results/individual_id_umap/stats_affinity_w25_h20_recsvd20_alpha2_hdbscan_all8_largest_working/zf_w25_h20_stats_pca512_recsvd20_a2_hdbscan_nn50_cosine/songmae_pool_stats_sliding_w25_h20_recsvd20_a2_pca_whiten_l2512_hdbscan_points.npz \
  --out_dir results/individual_id_latent_trajectory_decoder/smoke_zf \
  --epochs 1 \
  --probe_epochs 1 \
  --max_windows 128 \
  --batch_size 32 \
  --seq_len 16
```

zf pooled latent test:

```bash
python individual_id/latent_trajectory_decoder/train_probe.py \
  --umap_points_npz results/individual_id_latent_trajectory_decoder/zf_feature_caches/pooled_w25_h20_recsvd20_alpha2/songmae_pool_stats_sliding_w25_h20_recsvd20_a2_pca_whiten_l2512_features.npz \
  --out_dir results/individual_id_latent_trajectory_decoder/zf_pooled_latent_recsvd20_w25_h20_500k \
  --epochs 20 \
  --probe_epochs 200 \
  --batch_size 64 \
  --seq_len 32 \
  --stride 8 \
  --d_model 56
```

zf no-pooling stacked-latent test:

```bash
python individual_id/latent_trajectory_decoder/train_probe.py \
  --umap_points_npz results/individual_id_latent_trajectory_decoder/zf_feature_caches/unpooled_stacked_encoded_before/songmae_pool_mean_sliding_w1_h1_features.npz \
  --out_dir results/individual_id_latent_trajectory_decoder/zf_unpooled_stacked_latent_500k \
  --epochs 20 \
  --probe_epochs 200 \
  --max_windows 20000 \
  --batch_size 64 \
  --seq_len 32 \
  --stride 16 \
  --d_model 96
```
