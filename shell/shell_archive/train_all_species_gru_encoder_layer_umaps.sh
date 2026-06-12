#!/usr/bin/env bash
set -euo pipefail

ROOT="results/individual_id_latent_trajectory_decoder"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_UMAP_POINTS="${MAX_UMAP_POINTS:-25000}"

run_species() {
  local species="$1"
  local tag="$2"
  local features_npz="$3"

  local train_dir="${ROOT}/${species}_t2vec_gru_h1024_l2_bidir_keep050_cosine_${EPOCHS}epochs_${tag}"
  local umap_dir="${ROOT}/${species}_t2vec_gru_h1024_l2_bidir_keep050_encoder_layer2_unpooled_raw_samplewhole25k_${tag}"

  if [[ ! -f "${train_dir}/model_best.pt" ]]; then
    python individual_id/latent_trajectory_decoder/train_t2vec_gru_latents.py \
      --features_npz "${features_npz}" \
      --out_dir "${train_dir}" \
      --hidden_dim 1024 \
      --layers 2 \
      --cell gru \
      --bidirectional \
      --keep_prob 0.50 \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr 3e-4 \
      --val_fraction 0.2 \
      --seed 42
  fi

  if [[ ! -f "${umap_dir}/summary.json" ]]; then
    python individual_id/latent_trajectory_decoder/umap_t2vec_gru_encoder_layer.py \
      --features_npz "${features_npz}" \
      --model_pt "${train_dir}/model_best.pt" \
      --out_dir "${umap_dir}" \
      --species_display_name "${species}" \
      --encoder_layer 2 \
      --max_umap_points "${MAX_UMAP_POINTS}" \
      --umap_neighbors 50 \
      --umap_min_dist 0.1 \
      --umap_metric cosine \
      --umap_negative_sample_rate 5 \
      --silhouette_sample_size 10000 \
      --hdbscan_min_cluster_size 0 \
      --hdbscan_min_samples 10 \
      --seed 42
  fi
}

run_species "zf" "allsongs" "${ROOT}/zf_feature_caches/unpooled_patch_pre_pos_whiten_l2/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "bf" "30songs" "${ROOT}/bf_feature_caches/unpooled_patch_pre_pos_whiten_l2_30songs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "canary" "30songs" "${ROOT}/canary_feature_caches/unpooled_patch_pre_pos_whiten_l2_30songs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "chiffchaff" "30songs" "${ROOT}/chiffchaff_feature_caches/unpooled_patch_pre_pos_whiten_l2_30songs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "european_starling" "30songs" "${ROOT}/european_starling_feature_caches/unpooled_patch_pre_pos_whiten_l2_30songs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "little_owl" "30songs" "${ROOT}/little_owl_feature_caches/unpooled_patch_pre_pos_whiten_l2_30songs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "ovenbird" "allsongs" "${ROOT}/ovenbird_feature_caches/unpooled_patch_pre_pos_whiten_l2_allsongs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
run_species "tree_pipit" "30songs" "${ROOT}/tree_pipit_feature_caches/unpooled_patch_pre_pos_whiten_l2_30songs/songmae_patch_pre_pos_pool_mean_sliding_w1_h1_whiten_l21536_features.npz"
