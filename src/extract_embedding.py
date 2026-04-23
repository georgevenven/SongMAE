import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_loader import normalize_spectrogram_numpy, normalize_spectrogram_tensor
from audio2spec import compute_spectrogram
import aves
from individual_id.audio_augmentations import augment_audio_segment
from utils import load_audio_params, load_model_from_checkpoint


def fit_feature_postprocess(features, mode="none", dim=256):
    assert features.ndim == 2
    if mode == "none":
        return None

    assert mode in {"pca_whiten_l2", "whiten_l2"}
    if mode == "whiten_l2":
        assert features.shape[0] > 0
        std = features.std(axis=0)
        std = np.maximum(std, 1e-12)
        return {
            "kind": mode,
            "dim": int(features.shape[1]),
            "mean": features.mean(axis=0).astype(np.float32, copy=False),
            "std": std.astype(np.float32, copy=False),
        }

    assert features.shape[0] > 0
    n_components = min(int(dim), int(features.shape[0]), int(features.shape[1]))
    assert n_components > 0
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=0)
    pca.fit(features)
    return {
        "kind": mode,
        "dim": int(n_components),
        "mean": pca.mean_.astype(np.float32, copy=False),
        "components": pca.components_.astype(np.float32, copy=False),
        "explained_variance": pca.explained_variance_.astype(np.float32, copy=False),
    }


def save_feature_postprocess(path, transform):
    payload = {
        "kind": np.array(transform["kind"]),
        "dim": np.array(transform["dim"]),
        "mean": transform["mean"],
    }
    if transform["kind"] == "pca_whiten_l2":
        payload["components"] = transform["components"]
        payload["explained_variance"] = transform["explained_variance"]
    else:
        assert transform["kind"] == "whiten_l2"
        payload["std"] = transform["std"]
    np.savez(path, **payload)


def load_feature_postprocess(path):
    loaded = np.load(path)
    kind = str(loaded["kind"].item())
    assert kind in {"pca_whiten_l2", "whiten_l2"}
    transform = {
        "kind": kind,
        "dim": int(loaded["dim"].item()),
        "mean": loaded["mean"].astype(np.float32, copy=False),
    }
    if kind == "pca_whiten_l2":
        transform["components"] = loaded["components"].astype(np.float32, copy=False)
        transform["explained_variance"] = loaded["explained_variance"].astype(np.float32, copy=False)
    else:
        transform["std"] = loaded["std"].astype(np.float32, copy=False)
    return transform


def apply_feature_postprocess_transform(features, transform):
    if transform is None:
        return features.astype(np.float32, copy=False)

    centered = features.astype(np.float32, copy=False) - transform["mean"]
    if transform["kind"] == "pca_whiten_l2":
        projected = centered @ transform["components"].T
        scale = np.sqrt(np.maximum(transform["explained_variance"], 1e-12))
        whitened = (projected / scale).astype(np.float32, copy=False)
    else:
        assert transform["kind"] == "whiten_l2"
        whitened = (centered / np.maximum(transform["std"], 1e-12)).astype(np.float32, copy=False)
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (whitened / norms).astype(np.float32, copy=False)


def maybe_apply_feature_postprocess(
    features,
    mode="none",
    dim=256,
    load_path=None,
    save_path=None,
):
    if mode == "none":
        return features.astype(np.float32, copy=False), None

    assert mode in {"pca_whiten_l2", "whiten_l2"}
    if load_path is not None:
        transform = load_feature_postprocess(load_path)
    else:
        transform = fit_feature_postprocess(features, mode=mode, dim=dim)
        if save_path is not None:
            save_feature_postprocess(save_path, transform)
    return apply_feature_postprocess_transform(features, transform), transform


def maybe_postprocess_segments(
    segments,
    args,
    default_feature_key,
):
    mode = args.get("embedding_postprocess", "none")
    if mode == "none":
        return None

    feature_key = args.get("embedding_postprocess_key") or default_feature_key
    arrays = [segment[feature_key] for segment in segments if segment[feature_key].shape[0] > 0]
    if not arrays:
        return None

    stacked = np.concatenate(arrays, axis=0)
    transformed, transform = maybe_apply_feature_postprocess(
        stacked,
        mode=mode,
        dim=args.get("embedding_postprocess_dim", 256),
        load_path=args.get("embedding_postprocess_load"),
        save_path=args.get("embedding_postprocess_save"),
    )

    start = 0
    for segment in segments:
        length = int(segment[feature_key].shape[0])
        if length == 0:
            continue
        segment[feature_key] = transformed[start : start + length]
        start += length

    return {
        "mode": mode,
        "dim": int(transform["dim"]),
        "feature_key": feature_key,
        "load_path": args.get("embedding_postprocess_load"),
        "save_path": args.get("embedding_postprocess_save"),
    }


def ms_to_timebins(ms_value, audio_params):
    sr = audio_params[0]
    hop_size = audio_params[2]
    return int((ms_value / 1000) * sr / hop_size)


def load_json_events(json_path, audio_params, selected_bird=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    event_map = {}
    for rec in data.get("recordings", []):
        recording = rec.get("recording", {})
        bird_id = recording.get("bird_id", "")
        filename = recording.get("filename", "")
        stem = Path(filename).stem
        if selected_bird is not None:
            if bird_id != selected_bird:
                continue

        events = []
        for event in rec.get("detected_events", []):
            units = []
            for unit in event.get("units", []):
                units.append(
                    (
                        ms_to_timebins(unit["onset_ms"], audio_params),
                        ms_to_timebins(unit["offset_ms"], audio_params),
                        int(unit["id"]),
                    )
                )
            events.append(
                {
                    "on_timebins": ms_to_timebins(event["onset_ms"], audio_params),
                    "off_timebins": ms_to_timebins(event["offset_ms"], audio_params),
                    "units": units,
                }
            )
        event_map[stem] = events
    return event_map


def create_label_arr(event, start_timebin, end_timebin):
    labels = np.full((end_timebin - start_timebin,), fill_value=-1, dtype=np.int64)
    for start, end, unit_id in event["units"]:
        lo = max(int(start_timebin), int(start))
        hi = min(int(end) + 1, int(end_timebin))
        if lo >= hi:
            continue
        labels[lo - start_timebin : hi - start_timebin] = int(unit_id)
    return labels


def _resolve_recording_mode(args):
    mode = args.get("recording_mode")
    if mode is None:
        mode = "full_recordings" if args.get("full_recordings", False) else "events"
    assert mode in {"events", "full_recordings"}
    return mode


def _resolve_single_spec_path(spec_dir, recording_stem):
    spec_dir = Path(spec_dir)
    path = spec_dir / f"{recording_stem}.npy"
    if path.exists():
        return path

    paths = sorted(spec_dir.rglob(f"{recording_stem}.npy"))
    assert paths, f"Recording not found: {recording_stem}"
    assert len(paths) == 1, f"Multiple recordings found: {recording_stem}"
    return paths[0]


def _resolve_spec_paths(spec_dir, recording_stem, recording_stems=None):
    spec_dir = Path(spec_dir)
    if recording_stems is not None:
        return [_resolve_single_spec_path(spec_dir, stem) for stem in recording_stems]
    if recording_stem is None:
        paths = sorted(spec_dir.glob("*.npy"))
        if not paths:
            paths = sorted(spec_dir.rglob("*.npy"))
        assert paths
        return paths
    return [_resolve_single_spec_path(spec_dir, recording_stem)]


def _build_segment_from_audio(raw_segment, audio_params, patch_width):
    sr, n_mels, hop_size, fft = audio_params
    wav = raw_segment["audio"].detach().cpu().numpy().astype(np.float32, copy=False)
    spec = compute_spectrogram(
        wav,
        sr=sr,
        n_fft=fft,
        hop=hop_size,
        mel=True,
        n_mels=n_mels,
    ).astype(np.float32, copy=False)
    rounded_spec_length = spec.shape[1] - (spec.shape[1] % patch_width)
    if rounded_spec_length == 0:
        return None

    units = []
    for unit in raw_segment["labels_original"]:
        units.append(
            (
                ms_to_timebins(unit["onset_ms"], audio_params),
                ms_to_timebins(unit["offset_ms"], audio_params),
                int(unit["id"]),
            )
        )

    event = {
        "on_timebins": 0,
        "off_timebins": rounded_spec_length,
        "units": units,
    }
    return {
        "recording_stem": raw_segment["recording_stem"],
        "spectrogram": spec[:, :rounded_spec_length],
        "labels_original": create_label_arr(event, 0, rounded_spec_length),
    }


def load_recording_segments_from_audio(args, patch_width):
    spec_dir = Path(args["spec_dir"])
    audio = load_audio_params(spec_dir, require_stats=False)
    audio_params = (
        audio["sr"],
        audio["mels"],
        audio["hop_size"],
        audio["fft"],
    )
    raw = aves.load_recording_audio_segments(
        {
            "wav_root": args.get("wav_root"),
            "wav_manifest": args.get("wav_manifest"),
            "wav_exts": args.get("wav_exts"),
            "audio_sr": audio["sr"],
            "json_path": args.get("json_path"),
            "bird": args.get("bird"),
            "recording_stem": args.get("recording_stem"),
            "recording_stems": args.get("recording_stems"),
            "recording_mode": args.get("recording_mode"),
        }
    )

    segments = []
    for segment_index, raw_segment in enumerate(raw["segments"]):
        raw_segment = augment_audio_segment(raw_segment, args, segment_index)
        built = _build_segment_from_audio(raw_segment, audio_params, patch_width)
        if built is None:
            continue
        segments.append(built)

    return {
        "audio_params": audio_params,
        "segments": segments,
    }


def load_recording_segments(args, patch_width):
    spec_dir = Path(args["spec_dir"])
    audio = load_audio_params(spec_dir, require_stats=False)
    audio_params = (
        audio["sr"],
        audio["mels"],
        audio["hop_size"],
        audio["fft"],
    )
    event_map = {}
    json_path = args.get("json_path")
    if json_path:
        event_map = load_json_events(
            json_path,
            audio_params=audio_params,
            selected_bird=args.get("bird"),
        )

    recording_mode = _resolve_recording_mode(args)
    recording_stem = args.get("recording_stem")
    recording_stems = args.get("recording_stems")
    max_timebins = args.get("num_timebins")
    if max_timebins is not None:
        max_timebins = int(max_timebins)
        if max_timebins <= 0:
            max_timebins = None
    segments = []
    collected_timebins = 0

    paths = _resolve_spec_paths(spec_dir, recording_stem, recording_stems=recording_stems)
    if recording_stems is not None:
        allowed_stems = set(recording_stems)
        paths = [path for path in paths if path.stem in allowed_stems]
    if recording_stem is None and event_map:
        allowed_stems = set(event_map)
        paths = [path for path in paths if path.stem in allowed_stems]

    for path in paths:
        if max_timebins is not None and collected_timebins >= max_timebins:
            break
        stem = path.stem
        spec = np.load(path, mmap_mode="r")
        rounded_spec_length = spec.shape[1] - (spec.shape[1] % patch_width)
        if rounded_spec_length == 0:
            continue
        spec = np.array(spec[:, :rounded_spec_length], dtype=np.float32, copy=True)
        events = event_map.get(stem, [])

        if recording_mode == "full_recordings":
            units = []
            for event in events:
                units.extend(event["units"])
            selected_events = [
                {
                    "on_timebins": 0,
                    "off_timebins": rounded_spec_length,
                    "units": units,
                }
            ]
        else:
            selected_events = events

        if not selected_events:
            continue

        for event in selected_events:
            start = max(0, min(int(event["on_timebins"]), rounded_spec_length))
            end = max(start, min(int(event["off_timebins"]), rounded_spec_length))
            if start == end:
                continue
            if max_timebins is not None:
                remaining_timebins = max_timebins - collected_timebins
                if remaining_timebins <= 0:
                    break
                end = min(end, start + remaining_timebins)
                if start == end:
                    continue
            spec_segment = spec[:, start:end]
            labels_segment = create_label_arr(event, start, end)
            if spec_segment.shape[1] == 0:
                continue
            segments.append(
                {
                    "recording_stem": stem,
                    "spectrogram": spec_segment,
                    "labels_original": labels_segment,
                }
            )
            collected_timebins += spec_segment.shape[1]

    return {
        "audio_params": audio_params,
        "segments": segments,
    }


def _load_normalization_target_stats(args):
    stats_dir = args.get("normalization_stats_dir")
    if stats_dir is None:
        stats_dir = args["spec_dir"]
    audio = load_audio_params(stats_dir)
    return np.float32(audio["mean"]), np.float32(audio["std"])


def normalize_recording_segments(segments, mode, target_stats=None):
    if mode == "none":
        return segments

    assert mode in {
        "audio_params",
        "per_recording_cmvn",
        "per_recording_cmvn_rescaled_to_target_stats",
        "per_model_input_zscore",
    }
    if mode == "per_model_input_zscore":
        return segments
    if mode == "audio_params":
        assert target_stats is not None
        target_mean, target_std = target_stats
        normalized = []
        for segment in segments:
            spectrogram = normalize_spectrogram_numpy(
                segment["spectrogram"],
                "audio_params",
                mean=target_mean,
                std=target_std,
            )
            normalized.append(
                {
                    "recording_stem": segment["recording_stem"],
                    "spectrogram": spectrogram,
                    "labels_original": segment["labels_original"],
                }
            )
        return normalized
    specs = [segment["spectrogram"] for segment in segments if segment["spectrogram"].shape[1] > 0]
    if not specs:
        return segments
    recording = np.concatenate(specs, axis=1).astype(np.float32, copy=False)
    mean = recording.mean(axis=1, keepdims=True)
    std = recording.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-6)

    normalized = []
    for segment in segments:
        spectrogram = ((segment["spectrogram"] - mean) / std).astype(np.float32, copy=False)
        if mode == "per_recording_cmvn_rescaled_to_target_stats":
            assert target_stats is not None
            target_mean, target_std = target_stats
            spectrogram = (spectrogram * target_std + target_mean).astype(np.float32, copy=False)
        normalized.append(
            {
                "recording_stem": segment["recording_stem"],
                "spectrogram": spectrogram,
                "labels_original": segment["labels_original"],
            }
        )
    return normalized


def _extract_segment_arrays(
    spec_segment,
    labels_segment,
    model,
    device,
    model_num_timebins,
    patch_width,
    num_patches_height,
    num_patches_time,
    encoder_layer_idx,
    input_normalization_mode="none",
    minimal_output=False,
):
    spec_tensor = torch.from_numpy(spec_segment).unsqueeze(0).to(device)
    labels_tensor = torch.from_numpy(labels_segment).to(device)

    total_timebins = spec_tensor.shape[-1]
    batch_size = max(1, (total_timebins + model_num_timebins - 1) // model_num_timebins)
    padded_timebins = batch_size * model_num_timebins
    pad_amount = padded_timebins - total_timebins

    if pad_amount > 0:
        spec_tensor = F.pad(spec_tensor, (0, pad_amount), mode="constant", value=0)
        labels_tensor = F.pad(labels_tensor, (0, pad_amount), mode="constant", value=-1)

    _, mel, _ = spec_tensor.shape
    batched_spec = spec_tensor.reshape(1, mel, batch_size, model_num_timebins).permute(2, 0, 1, 3)
    if input_normalization_mode == "per_model_input_zscore":
        batched_spec = normalize_spectrogram_tensor(
            batched_spec,
            "per_file_zscore",
        )

    with torch.no_grad():
        patch_pre_pos = model.patch_projection(batched_spec)
        encoded, patch = model.forward_encoder_inference(
            batched_spec,
            encoder_layer_idx=encoder_layer_idx,
        )

    batch_count, hidden_dim, _, _ = encoded.permute(0, 2, 1).reshape(
        encoded.shape[0],
        encoded.shape[2],
        num_patches_height,
        num_patches_time,
    ).shape
    assert batch_count == batch_size
    assert hidden_dim == encoded.shape[2]

    encoded_grid = encoded.permute(0, 2, 1).reshape(batch_size, encoded.shape[2], num_patches_height, num_patches_time)
    patch_grid = patch.permute(0, 2, 1).reshape(batch_size, patch.shape[2], num_patches_height, num_patches_time)
    patch_pre_pos_grid = patch_pre_pos.reshape(batch_size, patch_pre_pos.shape[1], num_patches_height, num_patches_time)

    encoded_flat = encoded_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height * encoded.shape[2])
    patch_flat = patch_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height * patch.shape[2])
    patch_pre_pos_flat = patch_pre_pos_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height * patch_pre_pos.shape[1])
    spec_flat = batched_spec.squeeze(1).permute(0, 2, 1).reshape(-1, mel)
    pos_ids = torch.arange(0, num_patches_time, device=device).repeat(batch_size)

    if pad_amount > 0:
        pad_patches = pad_amount // patch_width
        if pad_patches > 0:
            encoded_flat = encoded_flat[:-pad_patches]
            patch_flat = patch_flat[:-pad_patches]
            patch_pre_pos_flat = patch_pre_pos_flat[:-pad_patches]
            pos_ids = pos_ids[:-pad_patches]
        spec_flat = spec_flat[:-pad_amount]
        labels_tensor = labels_tensor[:-pad_amount]

    if encoded_flat.shape[0] == 0:
        return None

    label_pool_pad = (-labels_tensor.numel()) % patch_width
    pooled_labels_input = labels_tensor
    if label_pool_pad > 0:
        pooled_labels_input = F.pad(pooled_labels_input, (0, label_pool_pad), mode="constant", value=-1)
    pooled_labels = F.max_pool1d(
        pooled_labels_input.float().view(1, 1, -1),
        kernel_size=patch_width,
        stride=patch_width,
    ).view(-1).long()
    out = {
        "encoded_before": encoded_flat.cpu().numpy().astype(np.float32, copy=False),
        "labels_downsampled": pooled_labels.cpu().numpy().astype(np.int64, copy=False),
    }
    if minimal_output:
        return out
    out["patch_pre_pos"] = patch_pre_pos_flat.cpu().numpy().astype(np.float32, copy=False)
    out["patch_before"] = patch_flat.cpu().numpy().astype(np.float32, copy=False)
    out["labels_original"] = labels_tensor.cpu().numpy().astype(np.int64, copy=False)
    out["spectrograms"] = spec_flat.cpu().numpy().astype(np.float32, copy=False)
    out["pos_ids"] = pos_ids.cpu().numpy().astype(np.int64, copy=False)
    return out


def load_model_state(args):
    model, config = load_model_from_checkpoint(
        run_dir=args["run_dir"],
        checkpoint_file=args.get("checkpoint"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    patch_height = int(config["patch_height"])
    patch_width = int(config["patch_width"])
    model_num_timebins = int(config["num_timebins"])
    num_patches_time = int(model_num_timebins / patch_width)
    num_patches_height = int(config["max_seq"] / num_patches_time)
    return {
        "model": model,
        "config": config,
        "run_dir": args["run_dir"],
        "device": device,
        "patch_height": patch_height,
        "patch_width": patch_width,
        "model_num_timebins": model_num_timebins,
        "num_patches_time": num_patches_time,
        "num_patches_height": num_patches_height,
    }


def get_native_input_normalization(model_state):
    config = model_state["config"]
    mode = config.get("input_normalization")
    if mode is None:
        has_audio_params = (Path(model_state["run_dir"]) / "audio_params.json").is_file()
        mode = "audio_params" if has_audio_params else "none"
    assert mode in {"none", "audio_params", "per_file_zscore"}
    if mode == "per_file_zscore":
        return "per_model_input_zscore", None
    if mode == "audio_params":
        return "audio_params", model_state["run_dir"]
    return "none", None


def extract_recording_embeddings_with_state(args, model_state):
    model = model_state["model"]
    config = model_state["config"]
    device = model_state["device"]
    patch_height = model_state["patch_height"]
    patch_width = model_state["patch_width"]
    model_num_timebins = model_state["model_num_timebins"]
    num_patches_time = model_state["num_patches_time"]
    num_patches_height = model_state["num_patches_height"]

    if float(args.get("train_audio_speed_max_pct", 0.0)) > 0.0:
        raw = load_recording_segments_from_audio(args, patch_width=patch_width)
    else:
        raw = load_recording_segments(args, patch_width=patch_width)
    normalization_mode = args.get("spec_normalization", "none")
    target_stats = None
    if normalization_mode in {"audio_params", "per_recording_cmvn_rescaled_to_target_stats"}:
        target_stats = _load_normalization_target_stats(args)
    raw_segments = normalize_recording_segments(
        raw["segments"],
        normalization_mode,
        target_stats=target_stats,
    )
    minimal_output = bool(args.get("minimal_output", False))
    segment_states = []

    for raw_segment in raw_segments:
        state = _extract_segment_arrays(
            spec_segment=raw_segment["spectrogram"],
            labels_segment=raw_segment["labels_original"],
            model=model,
            device=device,
            model_num_timebins=model_num_timebins,
            patch_width=patch_width,
            num_patches_height=num_patches_height,
            num_patches_time=num_patches_time,
            encoder_layer_idx=args.get("encoder_layer_idx"),
            input_normalization_mode=normalization_mode,
            minimal_output=minimal_output,
        )
        if state is None:
            continue
        state["recording_stem"] = raw_segment["recording_stem"]
        segment_states.append(state)

    if not segment_states:
        raise ValueError("No valid patches extracted for the requested recording set.")

    if minimal_output:
        segments = []
        for segment in segment_states:
            segments.append(
                {
                    "recording_stem": segment["recording_stem"],
                    "encoded_embeddings_before_pos_removal": segment["encoded_before"],
                    "labels_downsampled": segment["labels_downsampled"],
                }
            )
        feature_postprocess = maybe_postprocess_segments(
            segments,
            args,
            default_feature_key="encoded_embeddings_before_pos_removal",
        )
        return {
            "segments": segments,
            "audio_params": raw["audio_params"],
            "patch_height": patch_height,
            "patch_width": patch_width,
            "num_patches_time": num_patches_time,
            "num_patches_height": num_patches_height,
            "model_num_timebins": model_num_timebins,
            "mels": int(config["mels"]),
            "checkpoint": args.get("checkpoint") or "",
            "feature_postprocess": feature_postprocess,
        }

    max_pos = -1
    for segment in segment_states:
        if segment["pos_ids"].size == 0:
            continue
        max_pos = max(max_pos, int(segment["pos_ids"].max()))
    assert max_pos >= 0

    encoded_sums = np.zeros((max_pos + 1, segment_states[0]["encoded_before"].shape[1]), dtype=np.float64)
    patch_sums = np.zeros((max_pos + 1, segment_states[0]["patch_before"].shape[1]), dtype=np.float64)
    pos_counts = np.zeros((max_pos + 1,), dtype=np.int64)

    for segment in segment_states:
        pos_ids = segment["pos_ids"]
        encoded_before = segment["encoded_before"]
        patch_before = segment["patch_before"]
        unique_pos = np.unique(pos_ids)
        for pos in unique_pos:
            mask = pos_ids == pos
            encoded_sums[pos] += encoded_before[mask].sum(axis=0, dtype=np.float64)
            patch_sums[pos] += patch_before[mask].sum(axis=0, dtype=np.float64)
            pos_counts[pos] += int(mask.sum())

    valid_pos = pos_counts > 0
    assert np.any(valid_pos)
    encoded_means = np.zeros_like(encoded_sums, dtype=np.float32)
    patch_means = np.zeros_like(patch_sums, dtype=np.float32)
    encoded_means[valid_pos] = (encoded_sums[valid_pos] / pos_counts[valid_pos, None]).astype(np.float32, copy=False)
    patch_means[valid_pos] = (patch_sums[valid_pos] / pos_counts[valid_pos, None]).astype(np.float32, copy=False)

    segments = []
    for segment in segment_states:
        pos_ids = segment["pos_ids"]
        segments.append(
            {
                "recording_stem": segment["recording_stem"],
                "encoded_embeddings_before_pos_removal": segment["encoded_before"],
                "encoded_embeddings_after_pos_removal": (
                    segment["encoded_before"] - encoded_means[pos_ids]
                ).astype(np.float32, copy=False),
                "patch_embeddings_before_pos_encoding": segment["patch_pre_pos"],
                "patch_embeddings_before_pos_removal": segment["patch_before"],
                "patch_embeddings_after_pos_removal": (
                    segment["patch_before"] - patch_means[pos_ids]
                ).astype(np.float32, copy=False),
                "labels_original": segment["labels_original"],
                "labels_downsampled": segment["labels_downsampled"],
                "spectrograms": segment["spectrograms"],
                "pos_ids": pos_ids,
            }
        )

    feature_postprocess = maybe_postprocess_segments(
        segments,
        args,
        default_feature_key="encoded_embeddings_before_pos_removal",
    )

    return {
        "segments": segments,
        "audio_params": raw["audio_params"],
        "patch_height": patch_height,
        "patch_width": patch_width,
        "num_patches_time": num_patches_time,
        "num_patches_height": num_patches_height,
        "model_num_timebins": model_num_timebins,
        "mels": int(config["mels"]),
        "checkpoint": args.get("checkpoint") or "",
        "feature_postprocess": feature_postprocess,
    }


def extract_recording_embeddings(args):
    model_state = load_model_state(args)
    return extract_recording_embeddings_with_state(args, model_state)


def _concatenate_segments(segments, key):
    arrays = [segment[key] for segment in segments]
    assert arrays
    return np.concatenate(arrays, axis=0)


def main(args):
    extracted = extract_recording_embeddings(args)
    npz_path = args.get("npz_dir")
    if not npz_path:
        return extracted

    segments = extracted["segments"]
    np.savez(
        npz_path,
        spectrograms=_concatenate_segments(segments, "spectrograms"),
        labels_original=_concatenate_segments(segments, "labels_original"),
        labels_downsampled=_concatenate_segments(segments, "labels_downsampled"),
        encoded_embeddings_before_pos_removal=_concatenate_segments(
            segments,
            "encoded_embeddings_before_pos_removal",
        ),
        encoded_embeddings_after_pos_removal=_concatenate_segments(
            segments,
            "encoded_embeddings_after_pos_removal",
        ),
        patch_embeddings_before_pos_encoding=_concatenate_segments(
            segments,
            "patch_embeddings_before_pos_encoding",
        ),
        patch_embeddings_before_pos_removal=_concatenate_segments(
            segments,
            "patch_embeddings_before_pos_removal",
        ),
        patch_embeddings_after_pos_removal=_concatenate_segments(
            segments,
            "patch_embeddings_after_pos_removal",
        ),
        pos_ids=_concatenate_segments(segments, "pos_ids"),
        audio_sr=np.array(extracted["audio_params"][0]),
        audio_n_mels=np.array(extracted["audio_params"][1]),
        audio_hop_size=np.array(extracted["audio_params"][2]),
        audio_fft=np.array(extracted["audio_params"][3]),
        patch_height=np.array(extracted["patch_height"]),
        patch_width=np.array(extracted["patch_width"]),
        num_patches_time=np.array(extracted["num_patches_time"]),
        num_patches_height=np.array(extracted["num_patches_height"]),
        checkpoint=np.array(extracted["checkpoint"]),
        model_num_timebins=np.array(extracted["model_num_timebins"]),
        mels=np.array(extracted["mels"]),
        feature_postprocess_kind=np.array(
            (extracted.get("feature_postprocess") or {}).get("mode", "none")
        ),
        feature_postprocess_dim=np.array(
            int((extracted.get("feature_postprocess") or {}).get("dim", 0))
        ),
        feature_postprocess_key=np.array(
            (extracted.get("feature_postprocess") or {}).get("feature_key", "")
        ),
    )
    print(f"NPZ saved to {npz_path}")
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TinyBird embeddings for exact recordings or full directories.")
    parser.add_argument("--num_timebins", type=int, default=12400)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--spec_dir", type=str, required=True)
    parser.add_argument("--npz_dir", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--bird", type=str, default=None)
    parser.add_argument("--recording_stem", type=str, default=None)
    parser.add_argument("--embedding_postprocess", type=str, default="none", choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--embedding_postprocess_dim", type=int, default=256)
    parser.add_argument("--embedding_postprocess_key", type=str, default=None)
    parser.add_argument("--embedding_postprocess_load", type=str, default=None)
    parser.add_argument("--embedding_postprocess_save", type=str, default=None)
    parser.add_argument(
        "--recording_mode",
        type=str,
        default="events",
        choices=["events", "full_recordings"],
    )
    parser.add_argument(
        "--encoder_layer_idx",
        type=int,
        default=None,
        help="If set, extract embeddings from this encoder layer index.",
    )
    args = parser.parse_args()
    main(vars(args))
