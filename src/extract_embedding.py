import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils import load_audio_params, load_model_from_checkpoint


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


def _resolve_spec_paths(spec_dir, recording_stem):
    spec_dir = Path(spec_dir)
    if recording_stem is None:
        paths = sorted(spec_dir.glob("*.npy"))
        assert paths
        return paths

    path = spec_dir / f"{recording_stem}.npy"
    assert path.exists(), f"Recording not found: {recording_stem}"
    return [path]


def load_recording_segments(args, patch_width):
    spec_dir = Path(args["spec_dir"])
    audio = load_audio_params(spec_dir)
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
    max_timebins = args.get("num_timebins")
    if max_timebins is not None:
        max_timebins = int(max_timebins)
        if max_timebins <= 0:
            max_timebins = None
    segments = []
    collected_timebins = 0

    paths = _resolve_spec_paths(spec_dir, recording_stem)
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
            spectrogram = ((segment["spectrogram"] - target_mean) / target_std).astype(np.float32, copy=False)
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
        chunk_mean = batched_spec.mean(dim=(1, 2, 3), keepdim=True)
        chunk_std = batched_spec.std(dim=(1, 2, 3), keepdim=True)
        chunk_std = torch.clamp(chunk_std, min=1e-6)
        batched_spec = (batched_spec - chunk_mean) / chunk_std

    with torch.no_grad():
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

    encoded_flat = encoded_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height * encoded.shape[2])
    patch_flat = patch_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height * patch.shape[2])
    spec_flat = batched_spec.squeeze(1).permute(0, 2, 1).reshape(-1, mel)
    pos_ids = torch.arange(0, num_patches_time, device=device).repeat(batch_size)

    if pad_amount > 0:
        pad_patches = pad_amount // patch_width
        if pad_patches > 0:
            encoded_flat = encoded_flat[:-pad_patches]
            patch_flat = patch_flat[:-pad_patches]
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

    return {
        "encoded_before": encoded_flat.cpu().numpy().astype(np.float32, copy=False),
        "patch_before": patch_flat.cpu().numpy().astype(np.float32, copy=False),
        "labels_original": labels_tensor.cpu().numpy().astype(np.int64, copy=False),
        "labels_downsampled": pooled_labels.cpu().numpy().astype(np.int64, copy=False),
        "spectrograms": spec_flat.cpu().numpy().astype(np.float32, copy=False),
        "pos_ids": pos_ids.cpu().numpy().astype(np.int64, copy=False),
    }


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
        "device": device,
        "patch_height": patch_height,
        "patch_width": patch_width,
        "model_num_timebins": model_num_timebins,
        "num_patches_time": num_patches_time,
        "num_patches_height": num_patches_height,
    }


def extract_recording_embeddings_with_state(args, model_state):
    model = model_state["model"]
    config = model_state["config"]
    device = model_state["device"]
    patch_height = model_state["patch_height"]
    patch_width = model_state["patch_width"]
    model_num_timebins = model_state["model_num_timebins"]
    num_patches_time = model_state["num_patches_time"]
    num_patches_height = model_state["num_patches_height"]

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
        )
        if state is None:
            continue
        state["recording_stem"] = raw_segment["recording_stem"]
        segment_states.append(state)

    if not segment_states:
        raise ValueError("No valid patches extracted for the requested recording set.")

    encoded_all = np.concatenate([segment["encoded_before"] for segment in segment_states], axis=0)
    patch_all = np.concatenate([segment["patch_before"] for segment in segment_states], axis=0)
    pos_ids_all = np.concatenate([segment["pos_ids"] for segment in segment_states], axis=0)

    assert encoded_all.shape[0] > 0
    assert patch_all.shape[0] > 0
    assert pos_ids_all.shape[0] > 0

    unique_pos = np.unique(pos_ids_all)
    assert unique_pos.size > 0

    encoded_means = np.zeros((int(unique_pos.max()) + 1, encoded_all.shape[1]), dtype=np.float32)
    patch_means = np.zeros((int(unique_pos.max()) + 1, patch_all.shape[1]), dtype=np.float32)
    for pos in unique_pos:
        encoded_means[pos] = encoded_all[pos_ids_all == pos].mean(axis=0)
        patch_means[pos] = patch_all[pos_ids_all == pos].mean(axis=0)

    encoded_after_all = encoded_all - encoded_means[pos_ids_all]
    patch_after_all = patch_all - patch_means[pos_ids_all]

    start = 0
    segments = []
    for segment in segment_states:
        length = segment["encoded_before"].shape[0]
        end = start + length
        segments.append(
            {
                "recording_stem": segment["recording_stem"],
                "encoded_embeddings_before_pos_removal": segment["encoded_before"],
                "encoded_embeddings_after_pos_removal": encoded_after_all[start:end],
                "patch_embeddings_before_pos_removal": segment["patch_before"],
                "patch_embeddings_after_pos_removal": patch_after_all[start:end],
                "labels_original": segment["labels_original"],
                "labels_downsampled": segment["labels_downsampled"],
                "spectrograms": segment["spectrograms"],
                "pos_ids": segment["pos_ids"],
            }
        )
        start = end

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
