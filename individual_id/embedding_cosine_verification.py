#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import aves
import extract_embedding
import perch
from run_individual_id_umap import (
    _load_aves_segments,
    _load_recording_stems_by_bird,
    _load_spec_segments,
    _pick_recordings,
    _resolve_run_dir,
)


def _l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32, copy=False)


def _mean_rows(rows):
    assert rows.ndim == 2
    assert rows.shape[0] > 0
    return rows.mean(axis=0, keepdims=True).astype(np.float32, copy=False)


def _songmae_recording_embeddings(args, bird_id, recording_stem, model_state):
    try:
        extracted = extract_embedding.extract_recording_embeddings_with_state(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
                "spec_dir": str(args.spec_dir),
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
                "encoder_layer_idx": args.encoder_layer_idx,
                "spec_normalization": args.songmae_input_normalization,
                "normalization_stats_dir": args.songmae_input_normalization_stats_dir,
                "seed": args.seed,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "train_audio_speed_min_pct": args.train_audio_speed_min_pct,
                "train_audio_speed_max_pct": args.train_audio_speed_max_pct,
            },
            model_state,
        )
    except ValueError as exc:
        if str(exc) == "No valid patches extracted for the requested recording set.":
            return np.zeros((0, 0), dtype=np.float32)
        raise
    key = f"encoded_embeddings_{args.songmae_embedding_variant}_pos_removal"
    rows = []
    for segment in extracted["segments"]:
        features = segment[key]
        if features.shape[0] == 0:
            continue
        rows.append(_mean_rows(features))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return _l2_normalize(np.vstack(rows))


def _aves_recording_embeddings(args, bird_id, recording_stem, model_state):
    segments = _load_aves_segments(args, bird_id, recording_stem, model_state)
    rows = []
    for segment in segments:
        features = segment["features"]
        if features.shape[0] == 0:
            continue
        rows.append(_mean_rows(features))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return _l2_normalize(np.vstack(rows))


def _perch_recording_embeddings(args, bird_id, recording_stem, model_state):
    extracted = perch.extract_recording_embeddings_with_state(
        {
            "json_path": str(args.annotation_json),
            "bird": bird_id,
            "recording_stem": recording_stem,
            "recording_mode": args.recording_mode,
            "wav_root": args.wav_root,
            "wav_manifest": args.wav_manifest,
            "wav_exts": args.wav_exts,
            "perch_audio_sr": args.perch_audio_sr,
            "seed": args.seed,
            "train_audio_speed_min_pct": args.train_audio_speed_min_pct,
            "train_audio_speed_max_pct": args.train_audio_speed_max_pct,
        },
        model_state,
    )
    rows = []
    for segment in extracted["segments"]:
        features = segment["features"]
        if features.shape[0] == 0:
            continue
        rows.append(features.astype(np.float32, copy=False))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return _l2_normalize(np.vstack(rows))


def _spec_recording_embeddings(args, bird_id, recording_stem):
    segments = _load_spec_segments(args, bird_id, recording_stem, patch_width=1)
    rows = []
    for segment in segments:
        features = segment["features"].T.astype(np.float32, copy=False)
        if features.shape[0] == 0:
            continue
        rows.append(_mean_rows(features))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return _l2_normalize(np.vstack(rows))


def _extract_species_embeddings(args, model_state):
    stems_by_bird = _load_recording_stems_by_bird(args.annotation_json)
    bird_ids = sorted(stems_by_bird)
    embeddings = {}
    for bird_id in bird_ids:
        stems = _pick_recordings(
            stems_by_bird[bird_id],
            songs_per_bird=args.songs_per_bird,
            seed=args.seed,
            bird_id=bird_id,
        )
        if len(stems) < 2:
            continue
        recording_map = {}
        for stem in stems:
            if args.encoder == "SongMAE":
                rows = _songmae_recording_embeddings(args, bird_id, stem, model_state)
            elif args.encoder == "AVES":
                rows = _aves_recording_embeddings(args, bird_id, stem, model_state)
            elif args.encoder == "Spec":
                rows = _spec_recording_embeddings(args, bird_id, stem)
            else:
                assert args.encoder == "Perch"
                rows = _perch_recording_embeddings(args, bird_id, stem, model_state)
            if rows.shape[0] > 0:
                recording_map[stem] = rows
        if len(recording_map) >= 2:
            embeddings[bird_id] = recording_map
    return embeddings


def _sample_positive_score(rng, species_embeddings, birds):
    while True:
        bird_id = birds[int(rng.integers(len(birds)))]
        recording_map = species_embeddings[bird_id]
        stems = list(recording_map)
        if len(stems) < 2:
            continue
        i, j = rng.choice(len(stems), size=2, replace=False)
        left = recording_map[stems[int(i)]]
        right = recording_map[stems[int(j)]]
        a = left[int(rng.integers(left.shape[0]))]
        b = right[int(rng.integers(right.shape[0]))]
        return float(np.dot(a, b))


def _sample_negative_score(rng, species_embeddings, birds):
    i, j = rng.choice(len(birds), size=2, replace=False)
    left_map = species_embeddings[birds[int(i)]]
    right_map = species_embeddings[birds[int(j)]]
    left_stem = list(left_map)[int(rng.integers(len(left_map)))]
    right_stem = list(right_map)[int(rng.integers(len(right_map)))]
    left = left_map[left_stem]
    right = right_map[right_stem]
    a = left[int(rng.integers(left.shape[0]))]
    b = right[int(rng.integers(right.shape[0]))]
    return float(np.dot(a, b))


def _tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if valid.size == 0:
        return 0.0
    return float(tpr[valid[-1]])


def main():
    parser = argparse.ArgumentParser(description="Cosine-similarity verification on mean-pooled window embeddings.")
    parser.add_argument("--encoder", required=True, choices=["SongMAE", "Spec", "AVES", "Perch"])
    parser.add_argument("--species", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_budget", type=int, default=4000)
    parser.add_argument("--train_audio_speed_min_pct", type=float, default=0.0)
    parser.add_argument("--train_audio_speed_max_pct", type=float, default=0.0)
    parser.add_argument("--songmae_embedding_variant", default="before", choices=["before", "after"])
    parser.add_argument("--encoder_layer_idx", type=int, default=-1)
    parser.add_argument("--aves_model_path", default=None)
    parser.add_argument("--aves_config_path", default=None)
    parser.add_argument("--aves_audio_sr", type=int, default=16000)
    parser.add_argument("--audio_context_seconds", type=float, default=5.0)
    parser.add_argument("--wav_root", default=None)
    parser.add_argument("--wav_manifest", default=None)
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--perch_model_name", default="perch_v2")
    parser.add_argument("--perch_audio_sr", type=int, default=32000)
    parser.add_argument("--perch_window_seconds", type=float, default=5.0)
    parser.add_argument("--spec_normalization", default="none")
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    args = parser.parse_args()

    args.annotation_json = str(Path(args.annotation_json).resolve())
    args.spec_dir = str(Path(args.spec_dir).resolve())
    args.run_dir = str(_resolve_run_dir(args.run_dir))
    args.out_dir = str(Path(args.out_dir).resolve())
    if args.wav_root is not None:
        args.wav_root = str(Path(args.wav_root).resolve())
    if args.wav_manifest is not None:
        args.wav_manifest = str(Path(args.wav_manifest).resolve())
    if args.aves_model_path is not None:
        args.aves_model_path = str(Path(args.aves_model_path).resolve())
    if args.aves_config_path is not None:
        args.aves_config_path = str(Path(args.aves_config_path).resolve())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert 0.0 <= args.train_audio_speed_min_pct <= args.train_audio_speed_max_pct < 1.0

    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(
            {
                "run_dir": args.run_dir,
                "checkpoint": args.checkpoint,
            }
        )
        (
            args.songmae_input_normalization,
            args.songmae_input_normalization_stats_dir,
        ) = extract_embedding.get_native_input_normalization(model_state)
    elif args.encoder == "AVES":
        model_state = aves.load_model_state_for_inference(
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
        args.songmae_input_normalization = None
        args.songmae_input_normalization_stats_dir = None
    elif args.encoder == "Spec":
        model_state = None
        args.songmae_input_normalization = None
        args.songmae_input_normalization_stats_dir = None
    else:
        assert args.encoder == "Perch"
        model_state = perch.load_model_state_for_inference(
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
        args.songmae_input_normalization = None
        args.songmae_input_normalization_stats_dir = None

    species_embeddings = _extract_species_embeddings(args, model_state)
    birds = sorted(species_embeddings)
    assert len(birds) >= 2, "Need at least two individuals with at least two valid recordings each."

    rng = np.random.default_rng(args.seed)
    pos_scores = np.asarray(
        [_sample_positive_score(rng, species_embeddings, birds) for _ in range(args.pair_budget)],
        dtype=np.float32,
    )
    neg_scores = np.asarray(
        [_sample_negative_score(rng, species_embeddings, birds) for _ in range(args.pair_budget)],
        dtype=np.float32,
    )
    y_true = np.concatenate(
        [
            np.ones(args.pair_budget, dtype=np.int64),
            np.zeros(args.pair_budget, dtype=np.int64),
        ]
    )
    scores = np.concatenate([pos_scores, neg_scores]).astype(np.float32, copy=False)

    summary = {
        "encoder": args.encoder,
        "species": args.species,
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "songs_per_bird": int(args.songs_per_bird),
        "pair_budget": int(args.pair_budget),
        "train_audio_speed_min_pct": float(args.train_audio_speed_min_pct),
        "train_audio_speed_max_pct": float(args.train_audio_speed_max_pct),
        "num_birds": int(len(birds)),
        "num_recordings": int(sum(len(species_embeddings[bird]) for bird in birds)),
        "num_embeddings": int(
            sum(
                int(rows.shape[0])
                for bird in birds
                for rows in species_embeddings[bird].values()
            )
        ),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "tpr_at_fpr_1pct": _tpr_at_fpr(y_true, scores, 0.01),
        "tpr_at_fpr_5pct": _tpr_at_fpr(y_true, scores, 0.05),
        "per_bird_recordings": {bird: len(species_embeddings[bird]) for bird in birds},
        "per_bird_embeddings": {
            bird: int(sum(rows.shape[0] for rows in species_embeddings[bird].values()))
            for bird in birds
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "scores.npz",
        y_true=y_true,
        scores=scores,
        positive_scores=pos_scores,
        negative_scores=neg_scores,
    )
    print(
        f"[verification] encoder={args.encoder} species={args.species} "
        f"auc={summary['auc_roc']:.4f} tpr@1%={summary['tpr_at_fpr_1pct']:.4f} "
        f"embeddings={summary['num_embeddings']}"
    )


if __name__ == "__main__":
    main()
