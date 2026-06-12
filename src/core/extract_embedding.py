import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .data_loader import SpectrogramDatasetSupervised
from .utils import downsample_labels, normalize_spectrogram, timebins_to_ms
from .utils import load_audio_params, load_model_state

def load_recording_segments(args):
    max_timebins = args.get("num_timebins")
    if max_timebins is not None:
        max_timebins = int(max_timebins)
        if max_timebins <= 0:
            max_timebins = None

    ds = SpectrogramDatasetSupervised(
        args["spec_dir"],
        args.get("json_path"),
        n_timebins=None,
        recording_mode=args.get("recording_mode", "events"),
        recording_stem=args.get("recording_stem"),
        recording_stems=args.get("recording_stems"),
        selected_bird=args.get("bird"),
        normalize=False,
    )
    segments = []
    collected_timebins = 0
    audio_params = (ds.params.sr, ds.params.mels, ds.params.hop_size, ds.params.fft)
    for idx in range(len(ds)):
        if max_timebins is not None and collected_timebins >= max_timebins:
            break
        _, event = ds.samples[idx]
        spec, labels, stem = ds[idx]
        spec = spec.squeeze(0).numpy()
        labels = labels.numpy()
        start_timebin = 0 if event is None else int(event["on_timebins"])
        if max_timebins is not None:
            remaining = max_timebins - collected_timebins
            spec = spec[:, :remaining]
            labels = labels[:remaining]
        if spec.shape[1] == 0:
            continue
        end_timebin = start_timebin + spec.shape[1]
        segments.append(
            {
                "recording_stem": stem,
                "song_id": idx,
                "start_ms": timebins_to_ms(start_timebin, audio_params),
                "end_ms": timebins_to_ms(end_timebin, audio_params),
                "spectrogram": spec,
                "labels_original": labels,
            }
        )
        collected_timebins += spec.shape[1]

    return {
        "audio_params": audio_params,
        "segments": segments,
    }


def _load_normalization_stats(args, model_state):
    stats_dir = args.get("normalization_stats_dir") or model_state["run_dir"]
    audio = load_audio_params(stats_dir)
    return np.float32(audio["mean"]), np.float32(audio["std"])


def normalize_recording_segments(segments, mean, std):
    normalized = []
    for segment in segments:
        normalized.append(
            {
                "recording_stem": segment["recording_stem"],
                "song_id": segment["song_id"],
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "spectrogram": normalize_spectrogram(segment["spectrogram"], mean, std),
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
    with torch.no_grad():
        encoded, _ = model.forward_encoder_inference(
            batched_spec,
            encoder_layer_idx=encoder_layer_idx,
        )

    hidden_dim = encoded.shape[2]
    assert encoded.shape[0] == batch_size
    assert encoded.shape[1] == num_patches_height * num_patches_time

    encoded_grid = encoded.permute(0, 2, 1).reshape(batch_size, hidden_dim, num_patches_height, num_patches_time)
    encoded_grid = encoded_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height, hidden_dim)
    encoded_flat = encoded_grid.reshape(encoded_grid.shape[0], -1)
    spec_flat = batched_spec.squeeze(1).permute(0, 2, 1).reshape(-1, mel)

    if pad_amount > 0:
        pad_patches = pad_amount // patch_width
        if pad_patches > 0:
            encoded_grid = encoded_grid[:-pad_patches]
            encoded_flat = encoded_flat[:-pad_patches]
        spec_flat = spec_flat[:-pad_amount]
        labels_tensor = labels_tensor[:-pad_amount]

    if encoded_flat.shape[0] == 0:
        return None

    out = {
        "encoded_embeddings": encoded_flat.cpu().numpy().astype(np.float32, copy=False),
        "encoded_embeddings_grid": encoded_grid.cpu().numpy().astype(np.float32, copy=False),
        "labels_downsampled": downsample_labels(labels_tensor, encoded_flat.shape[0]),
        "labels_original": labels_tensor.cpu().numpy().astype(np.int64, copy=False),
        "spectrograms": spec_flat.cpu().numpy().astype(np.float32, copy=False),
    }
    return out


def extract_recording_embeddings_with_state(args, model_state):
    model = model_state["model"]
    config = model_state["config"]
    device = model_state["device"]
    patch_height = model_state["patch_height"]
    patch_width = model_state["patch_width"]
    model_num_timebins = model_state["model_num_timebins"]
    num_patches_time = model_state["num_patches_time"]
    num_patches_height = model_state["num_patches_height"]

    raw = load_recording_segments(args)
    mean, std = _load_normalization_stats(args, model_state)
    raw_segments = normalize_recording_segments(
        raw["segments"],
        mean,
        std,
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
        )
        if state is None:
            continue
        state["recording_stem"] = raw_segment["recording_stem"]
        state["song_id"] = raw_segment["song_id"]
        state["start_ms"] = raw_segment["start_ms"]
        state["end_ms"] = raw_segment["end_ms"]
        segment_states.append(state)

    if not segment_states:
        raise ValueError("No valid patches extracted for the requested recording set.")

    segments = []
    for segment in segment_states:
        segments.append(
            {
                "recording_stem": segment["recording_stem"],
                "song_id": segment["song_id"],
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "encoded_embeddings": segment["encoded_embeddings"],
                "encoded_embeddings_grid": segment["encoded_embeddings_grid"],
                "labels_original": segment["labels_original"],
                "labels_downsampled": segment["labels_downsampled"],
                "spectrograms": segment["spectrograms"],
            }
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
    }


def extract_recording_embeddings(args):
    model_state = load_model_state(args["run_dir"], args.get("checkpoint"), random_init=args.get("random_init", False))
    return extract_recording_embeddings_with_state(args, model_state)


def _concatenate_segments(segments, key):
    arrays = [segment[key] for segment in segments]
    assert arrays
    return np.concatenate(arrays, axis=0)


def _token_metadata(segments):
    stems, song_ids, starts, ends = [], [], [], []
    for segment in segments:
        count = segment["encoded_embeddings"].shape[0]
        edges = np.linspace(segment["start_ms"], segment["end_ms"], count + 1)
        stems.append(np.full(count, segment["recording_stem"]))
        song_ids.append(np.full(count, segment["song_id"], dtype=np.int64))
        starts.append(edges[:-1])
        ends.append(edges[1:])
    return np.concatenate(stems), np.concatenate(song_ids), np.concatenate(starts), np.concatenate(ends)


def main(args):
    extracted = extract_recording_embeddings(args)
    npz_path = args.get("npz_dir")
    if not npz_path:
        return extracted

    segments = extracted["segments"]
    recording_stem, song_id, token_start_ms, token_end_ms = _token_metadata(segments)
    np.savez(
        npz_path,
        spectrograms=_concatenate_segments(segments, "spectrograms"),
        labels_original=_concatenate_segments(segments, "labels_original"),
        labels_downsampled=_concatenate_segments(segments, "labels_downsampled"),
        encoded_embeddings=_concatenate_segments(segments, "encoded_embeddings"),
        encoded_embeddings_grid=_concatenate_segments(segments, "encoded_embeddings_grid"),
        recording_stem=recording_stem,
        song_id=song_id,
        token_start_ms=token_start_ms,
        token_end_ms=token_end_ms,
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
    parser = argparse.ArgumentParser(description="Extract SongMAE embeddings for exact recordings or full directories.")
    parser.add_argument("--num_timebins", type=int, default=12400)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--spec_dir", type=str, required=True)
    parser.add_argument("--npz_dir", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--bird", type=str, default=None)
    parser.add_argument("--random_init", action="store_true")
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
