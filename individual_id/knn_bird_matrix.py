#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.core import extract_embedding  # noqa: E402
from src.external_models import aves  # noqa: E402
from src.external_models import bird_mae  # noqa: E402
from src.external_models import hubert  # noqa: E402
try:
    from src.external_models import old_perch as perch  # noqa: E402
except ImportError:
    from src.external_models import perch2 as perch  # noqa: E402
from individual_identification_linear_probe import (  # noqa: E402
    _apply_spec_normalization_preset,
    _pool_recording,
)
from run_individual_id_umap import _load_patch_width  # noqa: E402

NAME_ALIASES = {
    "zf": "Zebra finch",
    "bf": "Bengalese finch",
    "canary": "Canary",
    "ovenbird": "Ovenbird",
    "chiffchaff": "Chiffchaff",
    "european_starling": "European starling",
    "tree_pipit": "Tree pipit",
    "little_owl": "Little owl",
}

KNN_CMAP = LinearSegmentedColormap.from_list("knn_overlap", ["#fffdf7", "#ffe66d", "#d7301f"])
KNN_NORM_GAMMA = 0.45
PURITY_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#332288",
    "#882255",
]

SPECIES = {
    "zf": ("Zebra_Finch", "files/annotation jsons/zf_annotations.json", "/media/george-vengrovski/disk2/specs/zf_64hop_32khz", "full_recordings"),
    "bf": ("bf", "files/annotation jsons/bf_annotations.json", "/media/george-vengrovski/disk2/specs/bf_64hop_32khz", "full_recordings"),
    "canary": ("canary", "files/annotation jsons/canary_annotations_for_individual_id.json", "/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz", "full_recordings"),
    "ovenbird": ("ovenbird", "files/annotation jsons/lapp_ovenbird.json", "/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz", "events"),
    "chiffchaff": ("chiffchaff", "files/annotation jsons/chiffchaff_annotations.json", "/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz", "full_recordings"),
    "european_starling": ("european_starling", "files/annotation jsons/european_starling_annotations.json", "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed", "full_recordings"),
    "tree_pipit": ("tree_pipit", "files/annotation jsons/tree_pipit_annotations.json", "/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz", "full_recordings"),
    "little_owl": ("little_owl", "files/annotation jsons/little_owl_annotations.json", "/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz", "full_recordings"),
}


def _parse_ints(text):
    values = sorted({int(x) for x in text.split(",") if x.strip()})
    assert values and min(values) > 0
    return values


def _subset_counts(text, max_count):
    if text == "all":
        return list(range(1, max_count + 1))
    return [x for x in _parse_ints(text) if x <= max_count]


def _resolve_run_dir(text):
    path = Path(text)
    if path.is_absolute() and path.is_dir():
        return path
    for base in [ROOT, ROOT / "runs"]:
        candidate = base / path
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(f"unable to resolve run_dir: {text}")


def _load_stems(annotation_json):
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    by_bird = {}
    for item in data["recordings"]:
        if not item.get("detected_events"):
            continue
        recording = item["recording"]
        bird_id = str(recording["bird_id"]).strip()
        stem = Path(recording["filename"]).stem
        by_bird.setdefault(bird_id, set()).add(stem)
    return {bird_id: sorted(stems) for bird_id, stems in by_bird.items()}


def _pick(stems, limit, seed, bird_id):
    if limit <= 0 or len(stems) <= limit:
        return list(stems)
    bird_seed = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + bird_seed)
    indices = np.sort(rng.choice(len(stems), size=limit, replace=False))
    return [stems[index] for index in indices]


def _cap_total_recordings(rows, limit, seed):
    if limit <= 0 or len(rows) <= limit:
        return rows
    by_bird = {}
    for row in rows:
        by_bird.setdefault(row["bird_id"], []).append(row)
    picked = []
    for bird_id, bird_rows in sorted(by_bird.items()):
        bird_limit = max(1, int(round(limit * len(bird_rows) / len(rows))))
        bird_seed = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + bird_seed)
        indices = np.sort(rng.choice(len(bird_rows), size=min(bird_limit, len(bird_rows)), replace=False))
        picked.extend(bird_rows[index] for index in indices)
    if len(picked) <= limit:
        return sorted(picked, key=lambda row: (row["bird_id"], row["recording_stem"]))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(picked), size=limit, replace=False))
    return sorted([picked[index] for index in indices], key=lambda row: (row["bird_id"], row["recording_stem"]))


def _point_cap_per_recording(args, recording_count):
    per_recording_cap = args.max_points_per_recording
    if args.max_total_points > 0:
        total_cap = max(1, args.max_total_points // recording_count)
        per_recording_cap = total_cap if per_recording_cap <= 0 else min(per_recording_cap, total_cap)
    return int(per_recording_cap)


def _cap_recording_features(args, row, features, recording_count):
    cap = _point_cap_per_recording(args, recording_count)
    if cap <= 0 or features.shape[0] <= cap:
        return features.astype(np.float32, copy=False)
    key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|bird-matrix"
    rng = np.random.default_rng(int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16))
    indices = np.sort(rng.choice(features.shape[0], size=cap, replace=False))
    return features[indices].astype(np.float32, copy=False)


def _feature_spill_path(args, row):
    key = f"{row['bird_id']}|{row['recording_stem']}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return Path(args.feature_memmap_dir) / f"{args.species_key}_recordings" / f"{digest}.npy"


def _store_recording_features(args, row, features):
    if not args.feature_memmap_dir:
        return {**row, "features": features}
    path = _feature_spill_path(args, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features.astype(np.float32, copy=False))
    return {**row, "feature_path": str(path), "feature_shape": tuple(features.shape)}


def _row_feature_shape(row):
    if "features" in row:
        return row["features"].shape
    return tuple(row["feature_shape"])


def _load_row_features(row):
    if "features" in row:
        return row["features"]
    return np.load(row["feature_path"], mmap_mode="r")


def _selected_recordings(args):
    species, annotation_json, spec_dir, recording_mode = SPECIES[args.species_key]
    args.species = species
    args.annotation_json = str(Path(args.annotation_json_override).resolve()) if args.annotation_json_override else str(ROOT / annotation_json)
    args.spec_dir = str(Path(args.spec_dir_override).resolve()) if args.spec_dir_override else spec_dir
    args.recording_mode = args.recording_mode_override or recording_mode
    args.run_dir = str(_resolve_run_dir(args.run_dir))

    rows = []
    for bird_id, stems in sorted(_load_stems(args.annotation_json).items()):
        if len(stems) < args.min_songs_per_bird:
            continue
        for stem in _pick(stems, args.songs_per_bird, args.seed, bird_id):
            rows.append({"bird_id": bird_id, "recording_stem": stem})
    assert rows, f"no singers have at least {args.min_songs_per_bird} recordings"
    return _cap_total_recordings(rows, args.max_recordings, args.seed)


def _feature_key(args):
    if args.embedding_variant == "before":
        return "encoded_embeddings_before_pos_removal"
    assert args.embedding_variant == "after"
    return "encoded_embeddings_after_pos_removal"


def _extract_songmae_tokens(args, selected):
    model_state = extract_embedding.load_model_state(args.run_dir, args.checkpoint)
    if args.spec_normalization == "auto":
        args.spec_normalization, args.normalization_stats_dir = "audio_params", model_state["run_dir"]
    key = _feature_key(args)
    request = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "spec_dir": args.spec_dir,
        "json_path": args.annotation_json,
        "recording_stems": [row["recording_stem"] for row in selected],
        "recording_mode": args.recording_mode,
        "encoder_layer_idx": args.encoder_layer_idx,
        "spec_normalization": args.spec_normalization,
        "normalization_stats_dir": args.normalization_stats_dir,
        "minimal_output": args.embedding_variant == "before",
        "embedding_postprocess": "none",
        "embedding_postprocess_dim": args.feature_postprocess_dim,
        "embedding_postprocess_key": key,
        "embedding_postprocess_load": None,
        "embedding_postprocess_save": None,
    }
    extracted = extract_embedding.extract_recording_embeddings_with_state(request, model_state)
    arrays_by_stem = {row["recording_stem"]: [] for row in selected}
    for segment in extracted["segments"]:
        x = segment[key].astype(np.float32, copy=False)
        if x.size:
            arrays_by_stem[segment["recording_stem"]].append(x)

    rows = []
    for row in selected:
        arrays = arrays_by_stem[row["recording_stem"]]
        if arrays:
            features = _cap_recording_features(args, row, np.vstack(arrays), len(selected))
            rows.append(_store_recording_features(args, row, features))
    assert rows
    return rows


def _load_encoder_state(args):
    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(args.run_dir, args.checkpoint)
        args.songmae_input_normalization = "audio_params"
        args.songmae_input_normalization_stats_dir = model_state["run_dir"]
        return model_state
    if args.encoder == "AVES":
        return aves.load_model_state_for_inference(
            {
                "run_dir": args.run_dir,
                "checkpoint": args.checkpoint,
                "encoder_layer_idx": args.encoder_layer_idx,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "audio_sr": args.aves_audio_sr,
                "audio_context_seconds": args.audio_context_seconds,
                "aves_model_path": args.aves_model_path,
                "aves_config_path": args.aves_config_path,
            }
        )
    if args.encoder == "Perch":
        return perch.load_model_state_for_inference(
            {
                "run_dir": args.run_dir,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "perch_model_name": args.perch_model_name,
                "perch_audio_sr": args.perch_audio_sr,
                "perch_window_seconds": args.perch_window_seconds,
            }
        )
    if args.encoder == "HuBERT":
        return hubert.load_model_state_for_inference(
            {
                "run_dir": args.run_dir,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "hubert_model_name": args.hubert_model_name,
                "hubert_audio_sr": args.hubert_audio_sr,
            }
        )
    if args.encoder == "BirdMAE":
        return bird_mae.load_model_state_for_inference(
            {
                "run_dir": args.run_dir,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "bird_mae_model_name": args.bird_mae_model_name,
                "bird_mae_audio_sr": args.bird_mae_audio_sr,
            }
        )
    assert args.encoder == "Spec"
    return None


def _extract_linear_encoder(args, selected):
    _apply_spec_normalization_preset(args)
    model_state = _load_encoder_state(args)
    patch_width = 1 if args.encoder in {"AVES", "Perch", "HuBERT", "BirdMAE"} else _load_patch_width(Path(args.run_dir))
    rows = []
    for row in selected:
        features, _, _ = _pool_recording(
            args,
            row["bird_id"],
            row["recording_stem"],
            patch_width,
            model_state,
            apply_audio_speed_augmentation=False,
        )
        if features.shape[0] > 0:
            features = _cap_recording_features(args, row, features, len(selected))
            rows.append(_store_recording_features(args, row, features))
    assert rows
    return rows


def _extract(args, selected):
    if args.encoder == "SongMAE" and args.songmae_affinity_features == "tokens":
        return _extract_songmae_tokens(args, selected)
    return _extract_linear_encoder(args, selected)


def _sample(args, rows):
    bird_ids = sorted({row["bird_id"] for row in rows})
    bird_to_code = {bird_id: index for index, bird_id in enumerate(bird_ids)}
    total_points = sum(_row_feature_shape(row)[0] for row in rows)
    feature_dim = _row_feature_shape(rows[0])[1]
    if args.feature_memmap_dir:
        feature_dir = Path(args.feature_memmap_dir)
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_path = feature_dir / f"{args.species_key}_features.npy"
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_points, feature_dim),
        )
        feature_storage = str(feature_path)
    else:
        features = np.empty((total_points, feature_dim), dtype=np.float32)
        feature_storage = "memory"
    point_birds = []
    point_recordings = []
    recording_birds = []
    recording_stems = []
    counts = []
    per_recording_cap = _point_cap_per_recording(args, len(rows))
    point_start = 0
    for recording_index, row in enumerate(rows):
        x = _load_row_features(row)
        if per_recording_cap > 0 and x.shape[0] > per_recording_cap:
            key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|bird-matrix"
            rng = np.random.default_rng(int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16))
            indices = np.sort(rng.choice(x.shape[0], size=per_recording_cap, replace=False))
            x = x[indices]
        bird = bird_to_code[row["bird_id"]]
        point_end = point_start + x.shape[0]
        features[point_start:point_end] = x
        row["features"] = None
        point_birds.extend([bird] * x.shape[0])
        point_recordings.extend([recording_index] * x.shape[0])
        recording_birds.append(bird)
        recording_stems.append(row["recording_stem"])
        counts.append(int(x.shape[0]))
        point_start = point_end
        if "feature_path" in row:
            Path(row["feature_path"]).unlink(missing_ok=True)
    return {
        "features": features,
        "point_birds": np.asarray(point_birds, dtype=np.int64),
        "point_recordings": np.asarray(point_recordings, dtype=np.int64),
        "recording_birds": np.asarray(recording_birds, dtype=np.int64),
        "recording_stems": np.asarray(recording_stems, dtype=object),
        "bird_ids": np.asarray(bird_ids, dtype=object),
        "sampled_counts": np.asarray(counts, dtype=np.int64),
        "point_cap_per_recording": int(per_recording_cap),
        "feature_storage": feature_storage,
    }


def _postprocess_sampled_features(args, sampled):
    if args.feature_postprocess == "none":
        sampled["features_l2_normalized"] = False
        return None
    if args.feature_postprocess == "pca_whiten_l2":
        features = sampled["features"]
        n_components = min(int(args.feature_postprocess_dim), int(features.shape[0]), int(features.shape[1]))
        assert n_components > 0
        fit_points = int(args.pca_fit_points)
        if fit_points > 0 and features.shape[0] > fit_points:
            rng = np.random.default_rng(args.seed)
            fit_indices = np.sort(rng.choice(features.shape[0], size=fit_points, replace=False))
            fit_features = features[fit_indices]
        else:
            fit_features = features
            fit_points = int(features.shape[0])
        mean = fit_features.mean(axis=0, dtype=np.float64)
        cov = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
        chunk_size = int(args.postprocess_chunk_size)
        for start in range(0, fit_features.shape[0], chunk_size):
            end = min(start + chunk_size, fit_features.shape[0])
            centered = fit_features[start:end].astype(np.float64, copy=False) - mean
            cov += centered.T @ centered
        cov /= max(fit_features.shape[0] - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1][:n_components]
        transform = {
            "kind": "pca_whiten_l2",
            "dim": int(n_components),
            "fit_points": int(fit_points),
            "mean": mean.astype(np.float32),
            "components": vectors[:, order].T.astype(np.float32),
            "explained_variance": values[order].astype(np.float32),
        }
        if n_components == features.shape[1]:
            out = features
            out_path = None
        elif sampled["feature_storage"] != "memory":
            out_path = Path(sampled["feature_storage"]).with_name(f"{args.species_key}_features_pca.npy")
            out = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(features.shape[0], n_components),
            )
        else:
            out = np.empty((features.shape[0], n_components), dtype=np.float32)
            out_path = None
        scale = np.sqrt(np.maximum(transform["explained_variance"], 1e-12))
        for start in range(0, features.shape[0], chunk_size):
            end = min(start + chunk_size, features.shape[0])
            centered = features[start:end].astype(np.float32, copy=False) - transform["mean"]
            projected = centered @ transform["components"].T
            projected = (projected / scale).astype(np.float32, copy=False)
            norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
            out[start:end] = projected / norms
        sampled["features"] = out
        sampled["features_l2_normalized"] = True
        if out_path is not None:
            Path(sampled["feature_storage"]).unlink(missing_ok=True)
            sampled["feature_storage"] = str(out_path)
        return transform
    transform = extract_embedding.fit_feature_postprocess(
        sampled["features"],
        mode=args.feature_postprocess,
        dim=args.feature_postprocess_dim,
    )
    sampled["features"] = extract_embedding.apply_feature_postprocess_transform(sampled["features"], transform)
    sampled["features_l2_normalized"] = True
    return transform


def _normalize_features_inplace(args, x):
    chunk_size = int(args.postprocess_chunk_size)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunk = x[start:end]
        norms = np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
        x[start:end] = chunk / norms


def _knn_full_cuda(args, sampled, k, device):
    x = sampled["features"]
    if not sampled.get("features_l2_normalized", False):
        _normalize_features_inplace(args, x)
    features = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    recordings = torch.from_numpy(sampled["point_recordings"]).to(device=device, dtype=torch.long)
    max_same_recording = int(torch.bincount(recordings).max().item())
    k = min(k, features.shape[0] - max_same_recording if args.exclude_same_recording else features.shape[0] - 1)
    assert k > 0

    neighbors = np.empty((features.shape[0], k), dtype=np.int64)
    arange = torch.arange(features.shape[0], device=device)
    for start in range(0, features.shape[0], args.knn_chunk_size):
        end = min(start + args.knn_chunk_size, features.shape[0])
        sims = features[start:end] @ features.T
        if args.exclude_same_recording:
            sims[recordings[start:end, None] == recordings[None, :]] = -float("inf")
        else:
            sims[torch.arange(end - start, device=device), arange[start:end]] = -float("inf")
        neighbors[start:end] = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    return neighbors, k


def _knn_blocked_cuda(args, sampled, k, device):
    x = sampled["features"]
    if not sampled.get("features_l2_normalized", False):
        _normalize_features_inplace(args, x)
    recordings_cpu = sampled["point_recordings"]
    recording_counts = np.bincount(recordings_cpu)
    max_same_recording = int(recording_counts.max())
    k = min(k, x.shape[0] - max_same_recording if args.exclude_same_recording else x.shape[0] - 1)
    assert k > 0

    neighbors = np.empty((x.shape[0], k), dtype=np.int64)
    candidate_chunk = int(args.knn_candidate_chunk_size)
    for start in range(0, x.shape[0], args.knn_chunk_size):
        end = min(start + args.knn_chunk_size, x.shape[0])
        query = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.float32)
        query_recordings = torch.from_numpy(recordings_cpu[start:end]).to(device=device, dtype=torch.long)
        best_scores = torch.full((end - start, k), -float("inf"), device=device)
        best_indices = torch.full((end - start, k), -1, device=device, dtype=torch.long)

        for cand_start in range(0, x.shape[0], candidate_chunk):
            cand_end = min(cand_start + candidate_chunk, x.shape[0])
            candidates = torch.from_numpy(x[cand_start:cand_end]).to(device=device, dtype=torch.float32)
            sims = query @ candidates.T
            if args.exclude_same_recording:
                cand_recordings = torch.from_numpy(recordings_cpu[cand_start:cand_end]).to(device=device, dtype=torch.long)
                sims[query_recordings[:, None] == cand_recordings[None, :]] = -float("inf")
            else:
                query_ids = torch.arange(start, end, device=device)
                cand_ids = torch.arange(cand_start, cand_end, device=device)
                sims[query_ids[:, None] == cand_ids[None, :]] = -float("inf")

            block_k = min(k, sims.shape[1])
            block_scores, block_indices = torch.topk(sims, k=block_k, dim=1)
            block_indices += cand_start
            merged_scores = torch.cat([best_scores, block_scores], dim=1)
            merged_indices = torch.cat([best_indices, block_indices], dim=1)
            best_scores, order = torch.topk(merged_scores, k=k, dim=1)
            best_indices = torch.gather(merged_indices, 1, order)

        neighbors[start:end] = best_indices.cpu().numpy()
    return neighbors, k


def _knn(args, sampled, k):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.knn_candidate_chunk_size > 0:
        neighbors, k = _knn_blocked_cuda(args, sampled, k, device)
    else:
        neighbors, k = _knn_full_cuda(args, sampled, k, device)
    return neighbors, str(device), k


def _purity(sampled, neighbors, k_values):
    labels = sampled["point_birds"]
    same = labels[neighbors] == labels[:, None]
    cumulative = np.cumsum(same, axis=1)
    return {k: float(np.mean(cumulative[:, k - 1] / k)) for k in k_values}


def _chance(sampled):
    labels = sampled["point_birds"]
    recordings = sampled["point_recordings"]
    bird_counts = np.bincount(labels)
    if not sampled["exclude_same_recording"]:
        return float(np.mean((bird_counts[labels] - 1) / max(labels.size - 1, 1)))
    recording_counts = np.bincount(recordings)
    same = bird_counts[labels] - recording_counts[recordings]
    total = labels.size - recording_counts[recordings]
    return float(np.mean(same / np.maximum(total, 1)))


def _bird_matrix(sampled, neighbors, k):
    labels = sampled["point_birds"]
    n_birds = sampled["bird_ids"].size
    matrix = np.zeros((n_birds, n_birds), dtype=np.float32)
    for bird in range(n_birds):
        query = labels == bird
        targets = labels[neighbors[query, :k]].reshape(-1)
        matrix[bird] = np.bincount(targets, minlength=n_birds)
        matrix[bird] /= max(float(query.sum() * k), 1.0)
    return matrix


def _recording_matrix(sampled, neighbors, k):
    recordings = sampled["point_recordings"]
    n_recordings = sampled["sampled_counts"].size
    matrix = np.zeros((n_recordings, n_recordings), dtype=np.float32)
    for recording in range(n_recordings):
        query = recordings == recording
        targets = recordings[neighbors[query, :k]].reshape(-1)
        matrix[recording] = np.bincount(targets, minlength=n_recordings)
        matrix[recording] /= max(float(query.sum() * k), 1.0)
    return matrix


def _stable_rank(matrix):
    graph = coo_matrix(matrix).tocsr()
    graph = (graph + graph.T) * 0.5
    graph.setdiag(0)
    graph.eliminate_zeros()
    return _stable_rank_graph(graph)


def _row_normalized_stable_rank(matrix):
    graph = coo_matrix(matrix).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    graph = graph.multiply(1.0 / row_sums[:, None]).tocsr()
    return _stable_rank_graph(graph)


def _stable_rank_graph(graph):
    if graph.nnz == 0:
        return 0.0
    frobenius_squared = float(graph.multiply(graph).sum())
    try:
        largest = float(svds(graph, k=1, return_singular_vectors=False, tol=1e-3)[0])
    except ValueError:
        largest = float(np.linalg.svd(graph.toarray(), compute_uv=False)[0])
    return frobenius_squared / max(largest**2, 1e-12)


def _linear_fit_r2(rows, prediction_key):
    y = np.asarray([row["true_count"] for row in rows], dtype=np.float64)
    x = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    if np.allclose(x, x[0]):
        return 0.0
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    denom = np.sum((y - y.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _subset_experiment(args, out_dir):
    path = out_dir / "knn_attribution_matrices.npz"
    assert path.exists(), f"run matrix build first: {path}"
    data = np.load(path, allow_pickle=True)
    matrix = data["recording_matrix"].astype(np.float32, copy=False)
    recording_birds = data["recording_birds"].astype(np.int64, copy=False)
    bird_ids = data["bird_ids"]
    counts = _subset_counts(args.subset_counts, bird_ids.size)
    recordings_per_bird = [int(x) for x in args.subset_recordings_per_bird.split(",") if x.strip()]
    if args.balanced_max_recordings_per_bird:
        recordings_per_bird = [-2]
    elif args.random_fraction_recordings_per_bird:
        recordings_per_bird = [-3]
    assert counts and recordings_per_bird

    rng = np.random.default_rng(args.seed)
    rows = []
    for true_count in counts:
        for per_bird in recordings_per_bird:
            for repeat in range(args.subset_repeats):
                birds = np.sort(rng.choice(bird_ids.size, size=true_count, replace=False))
                selected_recordings = [np.flatnonzero(recording_birds == bird) for bird in birds]
                if per_bird == -2:
                    n_balanced = min(x.size for x in selected_recordings)
                indices = []
                for bird_recordings in selected_recordings:
                    if per_bird == -2:
                        bird_recordings = np.sort(rng.choice(bird_recordings, size=n_balanced, replace=False))
                    elif per_bird == -3:
                        low = max(1, int(np.ceil(args.min_recording_fraction_per_bird * bird_recordings.size)))
                        n_recordings = rng.integers(low, bird_recordings.size + 1)
                        bird_recordings = np.sort(rng.choice(bird_recordings, size=n_recordings, replace=False))
                    elif per_bird > 0 and bird_recordings.size > per_bird:
                        bird_recordings = np.sort(rng.choice(bird_recordings, size=per_bird, replace=False))
                    indices.append(bird_recordings)
                indices = np.sort(np.concatenate(indices))
                subset = matrix[np.ix_(indices, indices)]
                rows.append(
                    {
                        "true_count": int(true_count),
                        "recordings_per_bird": int(per_bird),
                        "repeat": int(repeat),
                        "recordings": int(indices.size),
                        "stable_rank": _stable_rank(subset),
                        "row_normalized_stable_rank": _row_normalized_stable_rank(subset),
                    }
                )

    if args.random_fraction_recordings_per_bird:
        suffix = "_random_fraction"
    elif args.balanced_max_recordings_per_bird:
        suffix = "_balanced_max"
    else:
        suffix = "_all_recordings_all_counts"
    csv_path = out_dir / f"subset_recording_count_sweep{suffix}.csv"
    _write_rows_csv(csv_path, rows)
    by_recording_count = {}
    for per_bird in recordings_per_bird:
        group = [row for row in rows if row["recordings_per_bird"] == per_bird]
        by_recording_count[str(per_bird)] = {
            "stable_rank_r2": _linear_fit_r2(group, "stable_rank"),
            "row_normalized_stable_rank_r2": _linear_fit_r2(group, "row_normalized_stable_rank"),
            "rows": len(group),
        }
    summary = {
        "method": "recording_matrix_stable_rank_subset_count_sweep",
        "source": str(path),
        "subset_counts": counts,
        "subset_recordings_per_bird": recordings_per_bird,
        "subset_repeats": int(args.subset_repeats),
        "overall_stable_rank_r2": _linear_fit_r2(rows, "stable_rank"),
        "overall_row_normalized_stable_rank_r2": _linear_fit_r2(rows, "row_normalized_stable_rank"),
        "by_recordings_per_bird": by_recording_count,
    }
    (out_dir / f"subset_recording_count_sweep{suffix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _read_subset_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "true_count": int(row["true_count"]),
                    "recordings_per_bird": int(row["recordings_per_bird"]),
                    "repeat": int(row["repeat"]),
                    "recordings": int(row["recordings"]),
                    "stable_rank": float(row["stable_rank"]),
                    "row_normalized_stable_rank": float(row.get("row_normalized_stable_rank", row["stable_rank"])),
                }
            )
    assert rows
    return rows


def _linear_fit_xy(rows, x):
    y = np.asarray([row["true_count"] for row in rows], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    r2 = 1.0 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    return x, y, float(slope), float(intercept), float(r2)


def _plot_subset_experiment(args, out_dir):
    csv_path = Path(args.plot_subset_csv) if args.plot_subset_csv else out_dir / "subset_recording_count_sweep_all_recordings_all_counts.csv"
    rows = _read_subset_rows(csv_path)
    x_values = np.asarray([row[args.plot_prediction_key] for row in rows], dtype=np.float64)
    x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    ax.scatter(x, y, s=34, alpha=0.78, color="#2f6fbb", edgecolor="white", linewidth=0.45)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1.0)
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=12)
    ax.set_xlabel(args.plot_x_label)
    ax.set_ylabel("Known number of singers")
    ax.set_title(args.plot_title or f"{args.species_key} stable-rank count proxy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_base = Path(args.plot_out) if args.plot_out else out_dir / f"{csv_path.stem}_{args.plot_prediction_key}_regression"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    summary = {
        "source": str(csv_path),
        "png": str(out_base) + ".png",
        "pdf": str(out_base) + ".pdf",
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
        "prediction_key": args.plot_prediction_key,
        "rows": len(rows),
    }
    print(json.dumps(summary, indent=2))


def _panel_grid(items, figsize):
    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=300)
    return fig, list(zip(axes.flat, items))


def _save_all_species_heatmaps(root, key, filename):
    species_keys = list(NAME_ALIASES)
    fig, panels = _panel_grid(species_keys, (12, 6.8))
    axis_label = "Individual" if key == "bird_matrix" else "Recording"
    for ax, species_key in panels:
        data = np.load(root / species_key / "knn_attribution_matrices.npz", allow_pickle=True)
        matrix = data[key].astype(np.float32, copy=False)
        percentile = 97.5 if key == "bird_matrix" else 99.5
        vmax = max(float(np.percentile(matrix, percentile)), 1e-6)
        ax.imshow(matrix, cmap=KNN_CMAP, norm=PowerNorm(gamma=KNN_NORM_GAMMA, vmin=0.0, vmax=vmax))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(NAME_ALIASES[species_key], fontsize=15)
    for ax in fig.axes[::4]:
        ax.set_ylabel(axis_label, fontsize=14)
    for ax in fig.axes[4:]:
        ax.set_xlabel(axis_label, fontsize=14)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(root / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_all_species_scatter(root, prediction_key, x_label, filename,
                              csv_name="subset_recording_count_sweep_all_recordings_all_counts.csv"):
    species_keys = list(NAME_ALIASES)
    fig, panels = _panel_grid(species_keys, (12, 6.2))
    for ax, species_key in panels:
        path = root / species_key / csv_name
        if not path.exists():
            ax.set_axis_off()
            ax.set_title(f"{NAME_ALIASES[species_key]}\n(missing {csv_name})", fontsize=10)
            continue
        rows = _read_subset_rows(path)
        x_values = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
        x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)
        ax.scatter(x, y, s=14, alpha=0.72, color="#2f6fbb", edgecolor="white", linewidth=0.25)
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=0.8)
        ax.text(0.05, 0.94, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=11)
        ax.set_title(NAME_ALIASES[species_key], fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in fig.axes[::4]:
        ax.set_ylabel("Known singers", fontsize=14)
    for ax in fig.axes[4:]:
        ax.set_xlabel(x_label, fontsize=14)
    fig.tight_layout()
    fig.savefig(root / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _variable_recording_rows(root, species_key, min_fraction, repeats, seed):
    data = np.load(root / species_key / "knn_attribution_matrices.npz", allow_pickle=True)
    matrix = data["recording_matrix"].astype(np.float32, copy=False)
    recording_birds = data["recording_birds"].astype(np.int64, copy=False)
    n_birds = len(data["bird_ids"])
    by_bird = [np.flatnonzero(recording_birds == bird) for bird in range(n_birds)]
    max_recordings = min(len(x) for x in by_bird)
    min_recordings = max(1, int(np.ceil(min_fraction * max_recordings)))
    rng = np.random.default_rng(seed)
    rows = []
    for true_count in range(1, n_birds + 1):
        for recordings_per_bird in range(min_recordings, max_recordings + 1):
            for repeat in range(repeats):
                birds = np.sort(rng.choice(n_birds, size=true_count, replace=False))
                indices = []
                for bird in birds:
                    indices.append(rng.choice(by_bird[bird], size=recordings_per_bird, replace=False))
                indices = np.sort(np.concatenate(indices))
                subset = matrix[np.ix_(indices, indices)]
                rows.append(
                    {
                        "true_count": int(true_count),
                        "recordings_per_bird": int(recordings_per_bird),
                        "repeat": int(repeat),
                        "recordings": int(indices.size),
                        "stable_rank": _stable_rank(subset),
                        "row_normalized_stable_rank": _row_normalized_stable_rank(subset),
                    }
                )
    return rows, min_recordings, max_recordings


def _write_rows_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_stable_rank_rows(rows, key, title, out_base):
    x_values = np.asarray([row[key] for row in rows], dtype=np.float64)
    x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    ax.scatter(x, y, s=18, alpha=0.38, color="#2f6fbb", edgecolor="none")
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1.0)
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=12)
    ax.set_title(title)
    ax.set_xlabel("Stable rank")
    ax.set_ylabel("Known singers")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return r2


def _plot_variable_collage(root, species_keys, key, csv_stem, filename):
    fig, panels = _panel_grid(species_keys, (12, 6.2))
    for ax, species_key in panels:
        path = root / species_key / f"{csv_stem}.csv"
        rows = _read_subset_rows(path)
        x_values = np.asarray([row[key] for row in rows], dtype=np.float64)
        x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)
        ax.scatter(x, y, s=10, alpha=0.31, color="#2f6fbb", edgecolor="none")
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=0.8)
        ax.text(0.05, 0.94, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=11)
        ax.set_title(NAME_ALIASES[species_key], fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in fig.axes[::4]:
        ax.set_ylabel("Known singers", fontsize=14)
    for ax in fig.axes[4:]:
        ax.set_xlabel("Stable rank", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_variable_recording_stable_rank(root, min_fraction, repeats, seed):
    species_keys = list(NAME_ALIASES)
    summary = []
    stem = f"stable_rank_variable_recordings_per_bird_min{int(min_fraction * 100)}pct"
    for species_key in species_keys:
        rows, min_recordings, max_recordings = _variable_recording_rows(root, species_key, min_fraction, repeats, seed)
        _write_rows_csv(root / species_key / f"{stem}.csv", rows)
        raw_r2 = _plot_stable_rank_rows(rows, "stable_rank", NAME_ALIASES[species_key], root / species_key / stem)
        norm_r2 = _plot_stable_rank_rows(
            rows,
            "row_normalized_stable_rank",
            NAME_ALIASES[species_key],
            root / species_key / f"row_normalized_{stem}",
        )
        summary.append(
            {
                "species": species_key,
                "stable_rank_r2": raw_r2,
                "row_normalized_stable_rank_r2": norm_r2,
                "rows": len(rows),
                "min_recordings_per_bird": min_recordings,
                "max_recordings_per_bird": max_recordings,
            }
        )

    _plot_variable_collage(root, species_keys, "stable_rank", stem, f"all_species_{stem}")
    _plot_variable_collage(root, species_keys, "row_normalized_stable_rank", stem, f"all_species_row_normalized_{stem}")
    _write_rows_csv(root / f"{stem}_summary.csv", summary)


def _save_all_species_purity(root):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=300)
    for index, species_key in enumerate(NAME_ALIASES):
        path = root / species_key / "knn_purity.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        x = np.asarray([int(row["k"]) for row in rows], dtype=np.int64)
        y = np.asarray([float(row["purity"]) for row in rows], dtype=np.float64)
        ax.plot(
            x,
            y,
            color=PURITY_COLORS[index % len(PURITY_COLORS)],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            label=NAME_ALIASES[species_key],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x])
    ax.set_ylim(0, 1)
    ax.set_xlabel("k nearest neighbors", fontsize=16)
    ax.set_ylabel("Same-singer fraction", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_box_aspect(1)
    ax.legend(frameon=False, fontsize=10.5, loc="center left", bbox_to_anchor=(1.03, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "all_species_knn_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / "all_species_knn_purity.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_all_species_summary_csv(root):
    rows = []
    for species_key in NAME_ALIASES:
        summary = json.loads((root / species_key / "summary.json").read_text(encoding="utf-8"))
        subset = json.loads((root / species_key / "subset_recording_count_sweep_all_recordings_all_counts_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "species": species_key,
                "recordings": summary["recordings"],
                "points": summary["points"],
                "bird_diag_mean": summary["bird_diag_mean"],
                "bird_off_diag_mean": summary["bird_off_diag_mean"],
                "recording_diag_mean": summary["recording_diag_mean"],
                "recording_off_diag_mean": summary["recording_off_diag_mean"],
                "stable_rank_r2": subset["overall_stable_rank_r2"],
                "row_normalized_stable_rank_r2": subset["overall_row_normalized_stable_rank_r2"],
            }
        )
    with (root / "all_species_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_all_species_summary(args):
    root = Path(args.out_dir)
    _save_all_species_heatmaps(root, "bird_matrix", "all_species_bird_knn_heatmaps")
    _save_all_species_heatmaps(root, "recording_matrix", "all_species_recording_knn_heatmaps")
    _save_all_species_scatter(root, "stable_rank", "Stable rank", "all_species_stable_rank_proxy")
    _save_all_species_scatter(
        root,
        "row_normalized_stable_rank",
        "Row-normalized stable rank",
        "all_species_row_normalized_stable_rank_proxy",
    )
    _save_all_species_scatter(
        root,
        "stable_rank",
        "Stable rank",
        "all_species_stable_rank_random_fraction_proxy",
        csv_name="subset_recording_count_sweep_random_fraction.csv",
    )
    _save_all_species_scatter(
        root,
        "row_normalized_stable_rank",
        "Row-normalized stable rank",
        "all_species_row_normalized_stable_rank_random_fraction_proxy",
        csv_name="subset_recording_count_sweep_random_fraction.csv",
    )
    _save_variable_recording_stable_rank(root, 0.30, args.variable_recording_repeats, args.seed)
    _save_all_species_purity(root)
    _save_all_species_summary_csv(root)
    print(
        json.dumps(
            {
                "out_dir": str(root),
                "figures": [
                    "all_species_bird_knn_heatmaps.png",
                    "all_species_recording_knn_heatmaps.png",
                    "all_species_stable_rank_proxy.png",
                    "all_species_row_normalized_stable_rank_proxy.png",
                    "all_species_stable_rank_random_fraction_proxy.png",
                    "all_species_row_normalized_stable_rank_random_fraction_proxy.png",
                    "all_species_stable_rank_variable_recordings_per_bird_min30pct.png",
                    "all_species_row_normalized_stable_rank_variable_recordings_per_bird_min30pct.png",
                    "all_species_knn_purity.png",
                    "all_species_summary.csv",
                ],
            },
            indent=2,
        )
    )


def _save_purity_plot(args, out_dir, k_values, purity, chance):
    x = np.asarray(k_values, dtype=np.int64)
    y = np.asarray([purity[k] for k in k_values], dtype=np.float64)
    species_index = list(NAME_ALIASES).index(args.species_key)
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=300)
    ax.plot(
        x,
        y,
        color=PURITY_COLORS[species_index % len(PURITY_COLORS)],
        marker="o",
        markersize=5.8,
        linewidth=2.0,
    )
    ax.axhline(chance, color="#4D4D4D", linestyle="--", linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x])
    ax.set_ylim(0.0, min(1.0, max(0.2, float(y.max()) + 0.08)))
    ax.set_xlabel("k nearest neighbors", fontsize=16)
    ax.set_ylabel("Same-singer fraction", fontsize=16)
    ax.set_title(NAME_ALIASES[args.species_key], fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "knn_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "knn_purity.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_bird_heatmap(args, out_dir, matrix, bird_ids):
    fig_size = max(5.0, min(12.0, 0.35 * len(bird_ids)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    vmax = max(float(np.percentile(matrix, 97.5)), 1e-6)
    ax.imshow(matrix, cmap=KNN_CMAP, norm=PowerNorm(gamma=KNN_NORM_GAMMA, vmin=0.0, vmax=vmax))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Neighbor recording", fontsize=24)
    ax.set_ylabel("Query", fontsize=24)
    ax.set_title(NAME_ALIASES[args.species_key], fontsize=24)
    fig.tight_layout()
    fig.savefig(out_dir / "bird_knn_attribution.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "bird_knn_attribution.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_recording_heatmap(args, out_dir, matrix):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    vmax = max(float(np.percentile(matrix, 99.5)), 1e-6)
    ax.imshow(matrix, cmap=KNN_CMAP, norm=PowerNorm(gamma=KNN_NORM_GAMMA, vmin=0.0, vmax=vmax))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Neighbor recording", fontsize=24)
    ax.set_ylabel("Query", fontsize=24)
    ax.set_title(NAME_ALIASES[args.species_key], fontsize=24)
    fig.tight_layout()
    fig.savefig(out_dir / "recording_knn_attribution.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "recording_knn_attribution.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_outputs(args, sampled, neighbors, device, actual_k, out_dir):
    k_values = [k for k in _parse_ints(args.k_values) if k <= actual_k]
    matrix_k = min(args.matrix_k, actual_k)
    sampled["exclude_same_recording"] = args.exclude_same_recording
    purity = _purity(sampled, neighbors, k_values)
    chance = _chance(sampled)
    bird_matrix = _bird_matrix(sampled, neighbors, matrix_k)
    recording_matrix = _recording_matrix(sampled, neighbors, matrix_k)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_purity_plot(args, out_dir, k_values, purity, chance)
    _save_bird_heatmap(args, out_dir, bird_matrix, sampled["bird_ids"])
    _save_recording_heatmap(args, out_dir, recording_matrix)

    np.savez_compressed(
        out_dir / "knn_attribution_matrices.npz",
        k_values=np.asarray(k_values, dtype=np.int64),
        purity=np.asarray([purity[k] for k in k_values], dtype=np.float32),
        bird_matrix=bird_matrix,
        recording_matrix=recording_matrix,
        bird_ids=sampled["bird_ids"],
        recording_birds=sampled["recording_birds"],
        recording_stems=sampled["recording_stems"],
    )
    with (out_dir / "knn_purity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "purity", "chance"])
        writer.writeheader()
        for k in k_values:
            writer.writerow({"k": k, "purity": purity[k], "chance": chance})
    summary = {
        "species_key": args.species_key,
        "encoder": args.encoder,
        "songmae_affinity_features": args.songmae_affinity_features,
        "feature_postprocess": args.feature_postprocess,
        "feature_postprocess_dim": int(args.feature_postprocess_dim),
        "pca_fit_points": int(args.pca_fit_points),
        "recording_filter": "detected_events_nonempty",
        "device": device,
        "feature_storage": sampled["feature_storage"],
        "recordings": int(sampled["sampled_counts"].size),
        "points": int(sampled["features"].shape[0]),
        "max_recordings": int(args.max_recordings),
        "max_points_per_recording": int(args.max_points_per_recording),
        "max_total_points": int(args.max_total_points),
        "point_cap_per_recording": int(sampled["point_cap_per_recording"]),
        "songs_per_bird": int(args.songs_per_bird),
        "pool_window": int(args.pool_window),
        "pool_hop": int(args.pool_hop),
        "pool_mode": args.pool_mode,
        "matrix_k": int(matrix_k),
        "chance": chance,
        "purity": {str(k): purity[k] for k in k_values},
        "bird_diag_mean": float(np.diag(bird_matrix).mean()),
        "bird_off_diag_mean": float(bird_matrix[~np.eye(bird_matrix.shape[0], dtype=bool)].mean()),
        "recording_diag_mean": float(np.diag(recording_matrix).mean()),
        "recording_off_diag_mean": float(recording_matrix[~np.eye(recording_matrix.shape[0], dtype=bool)].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build bird/recording kNN attribution matrices.")
    parser.add_argument("species_key", choices=sorted(SPECIES))
    parser.add_argument("--encoder", default="SongMAE", choices=["SongMAE", "Spec", "AVES", "Perch", "HuBERT", "BirdMAE"])
    parser.add_argument("--run_dir", default=str(ROOT / "github_assets/xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8"))
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "individual_id_knn_graph_metrics" / "bird_knn_matrix"))
    parser.add_argument("--songs_per_bird", type=int, default=0)
    parser.add_argument("--min_songs_per_bird", type=int, default=0)
    parser.add_argument("--max_recordings", type=int, default=0)
    parser.add_argument("--max_points_per_recording", type=int, default=400)
    parser.add_argument("--max_total_points", type=int, default=50000)
    parser.add_argument("--feature_memmap_dir", default=None)
    parser.add_argument("--k_values", default="1,2,5,10,20,50,100")
    parser.add_argument("--matrix_k", type=int, default=50)
    parser.add_argument("--postprocess_chunk_size", type=int, default=65536)
    parser.add_argument("--pca_fit_points", type=int, default=0)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--knn_candidate_chunk_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_experiment", action="store_true")
    parser.add_argument("--subset_counts", default="all")
    parser.add_argument("--subset_recordings_per_bird", default="10,20,40,80,0")
    parser.add_argument("--subset_repeats", type=int, default=3)
    parser.add_argument("--balanced_max_recordings_per_bird", action="store_true")
    parser.add_argument("--random_fraction_recordings_per_bird", action="store_true")
    parser.add_argument("--min_recording_fraction_per_bird", type=float, default=0.10)
    parser.add_argument("--plot_subset_experiment", action="store_true")
    parser.add_argument("--plot_subset_csv", default=None)
    parser.add_argument("--plot_out", default=None)
    parser.add_argument("--plot_title", default=None)
    parser.add_argument("--plot_prediction_key", default="stable_rank", choices=["stable_rank", "row_normalized_stable_rank"])
    parser.add_argument("--plot_x_label", default="Stable rank")
    parser.add_argument("--plot_all_species_summary", action="store_true")
    parser.add_argument("--variable_recording_repeats", type=int, default=5)
    parser.add_argument("--exclude_same_recording", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--embedding_variant", default="before", choices=["before", "after"])
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--songmae_affinity_features", default="tokens", choices=["tokens", "linear_probe"])
    parser.add_argument("--pool_window", type=int, default=30)
    parser.add_argument("--pool_hop", type=int, default=30)
    parser.add_argument("--pool_mode", default="mean", choices=["mean"])
    parser.add_argument("--window_mean_pool", action="store_true")
    parser.add_argument("--window_concat_pool", action="store_true")
    parser.add_argument("--window_token_probe", action="store_true")
    parser.add_argument("--feature_postprocess", default="pca_whiten_l2", choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--feature_postprocess_dim", type=int, default=1024)
    parser.add_argument("--spec_normalization", default="auto")
    parser.add_argument("--normalization_preset", choices=["vanilla", "zscore", "zscore_rescaled"], default=None)
    parser.add_argument("--audio_params_stats_dir", default=None)
    parser.add_argument("--normalization_stats_dir", default=None)
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    parser.add_argument("--annotation_json_override", default=None)
    parser.add_argument("--spec_dir_override", default=None)
    parser.add_argument("--recording_mode_override", default=None, choices=["events", "full_recordings"])
    parser.add_argument("--songmae_embedding_variant", choices=["before", "after"], default="before")
    parser.add_argument("--aves_model_path", default=None)
    parser.add_argument("--aves_config_path", default=None)
    parser.add_argument("--wav_root", default=None)
    parser.add_argument("--wav_manifest", default=None)
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--aves_audio_sr", type=int, default=16000)
    parser.add_argument("--perch_model_name", default="perch_v2")
    parser.add_argument("--perch_audio_sr", type=int, default=32000)
    parser.add_argument("--perch_window_seconds", type=float, default=5.0)
    parser.add_argument("--hubert_model_name", default="facebook/hubert-base-ls960")
    parser.add_argument("--hubert_audio_sr", type=int, default=16000)
    parser.add_argument("--bird_mae_model_name", default="DBD-research-group/Bird-MAE-Base")
    parser.add_argument("--bird_mae_audio_sr", type=int, default=32000)
    parser.add_argument("--audio_context_seconds", type=float, default=2.0)
    parser.add_argument("--train_audio_speed_min_pct", type=float, default=0.0)
    parser.add_argument("--train_audio_speed_max_pct", type=float, default=0.0)
    args = parser.parse_args()

    if args.encoder != "SongMAE" and args.spec_normalization == "auto":
        args.spec_normalization = "none"
    if args.audio_params_stats_dir is not None:
        args.audio_params_stats_dir = str(Path(args.audio_params_stats_dir).resolve())
    if args.spec_normalization_stats_dir is not None:
        args.spec_normalization_stats_dir = str(Path(args.spec_normalization_stats_dir).resolve())
    if args.aves_model_path is not None:
        args.aves_model_path = str(Path(args.aves_model_path).resolve())
    if args.aves_config_path is not None:
        args.aves_config_path = str(Path(args.aves_config_path).resolve())
    if args.wav_root is not None:
        args.wav_root = str(Path(args.wav_root).resolve())
    if args.wav_manifest is not None:
        args.wav_manifest = str(Path(args.wav_manifest).resolve())
    args.songmae_input_normalization = None
    args.songmae_input_normalization_stats_dir = None

    out_dir = Path(args.out_dir) / args.species_key
    if args.subset_experiment:
        _subset_experiment(args, out_dir)
        return
    if args.plot_subset_experiment:
        _plot_subset_experiment(args, out_dir)
        return
    if args.plot_all_species_summary:
        _plot_all_species_summary(args)
        return

    selected = _selected_recordings(args)
    rows = _extract(args, selected)
    sampled = _sample(args, rows)
    _postprocess_sampled_features(args, sampled)
    neighbors, device, actual_k = _knn(args, sampled, max(max(_parse_ints(args.k_values)), args.matrix_k))
    _write_outputs(args, sampled, neighbors, device, actual_k, out_dir)


if __name__ == "__main__":
    main()
