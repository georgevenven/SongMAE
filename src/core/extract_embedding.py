import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .data_loader import SpectrogramDatasetSupervised
from .embedding_store import EmbeddingFolderWriter
from .model import TARGET_FEATURE_TYPES
from .utils import create_label_arr, load_spec, normalize_spectrogram, patch_labels, timebins_to_ms
from .utils import load_model_state


def _labels_for_recording(ds, stem, length):
    units = [unit for event in ds.events_by_file[stem] for unit in event["units"]]
    return create_label_arr({"units": units}, 0, length)


def load_recording_segments(args):
    max_timebins = int(args.get("num_timebins") or 0)

    ds = SpectrogramDatasetSupervised(
        args["spec_dir"],
        args.get("json_path"),
        n_timebins=None,
        recording_mode=args.get("recording_mode", "events"),
        recording_stem=args.get("recording_stem"),
        recording_stems=args.get("recording_stems"),
        selected_bird=args.get("bird"),
    )
    segments = []
    collected_timebins = 0
    audio_params = (ds.params.sr, ds.params.mels, ds.params.hop_size, ds.params.fft)
    max_segments = int(args.get("max_segments") or 0)
    cached_path = None
    cached_spec = None
    cached_labels = None
    for idx in range(len(ds)):
        if max_segments > 0 and len(segments) >= max_segments:
            break
        if max_timebins > 0 and collected_timebins >= max_timebins:
            break
        spec_path, event = ds.samples[idx]
        if spec_path != cached_path:
            cached_path = spec_path
            cached_spec = load_spec(spec_path)
            cached_labels = _labels_for_recording(ds, spec_path.stem, cached_spec.shape[1])
        assert cached_spec is not None and cached_labels is not None

        spec, start_timebin, end_timebin = ds._event_crop(cached_spec, event) if event else ds._crop_pad(cached_spec)
        labels = cached_labels[start_timebin:end_timebin]
        if ds.normalize:
            spec = normalize_spectrogram(spec, ds.mean, ds.std)
        if max_timebins > 0:
            remaining = max_timebins - collected_timebins
            spec = spec[:, :remaining]
            labels = labels[:remaining]
        if spec.shape[1] == 0:
            continue
        end_timebin = start_timebin + spec.shape[1]
        segments.append(
            {
                "recording_stem": spec_path.stem,
                "song_id": idx,
                "start_ms": timebins_to_ms(start_timebin, audio_params),
                "end_ms": timebins_to_ms(end_timebin, audio_params),
                "spec_path": str(spec_path),
                "spectrogram": spec,
                "labels_original": labels,
            }
        )
        collected_timebins += spec.shape[1]

    return {
        "audio_params": audio_params,
        "segments": segments,
    }


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
    target_feature_type,
    all_layers,
):
    assert not (all_layers and encoder_layer_idx is not None)
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
    valid_timebins = torch.full((batch_size,), model_num_timebins, dtype=torch.long, device=device)
    if pad_amount > 0:
        valid_timebins[-1] = model_num_timebins - pad_amount
    with torch.no_grad():
        encoded, _ = model.forward_encoder_inference(
            batched_spec,
            encoder_layer_idx=encoder_layer_idx,
            target_feature_type=target_feature_type,
            valid_timebins=valid_timebins,
            return_all_layers=all_layers,
        )

    assert encoded.shape[0] == batch_size
    if all_layers:
        _, _, tokens, hidden_dim = encoded.shape
        assert tokens == num_patches_height * num_patches_time
        encoded_grid = encoded.reshape(
            batch_size, encoded.shape[1], num_patches_height, num_patches_time, hidden_dim
        )
        encoded_grid = encoded_grid.permute(0, 3, 1, 2, 4).reshape(
            -1, encoded.shape[1], num_patches_height, hidden_dim
        )
        encoded_flat = encoded_grid.flatten(2)
    else:
        hidden_dim = encoded.shape[2]
        assert encoded.shape[1] == num_patches_height * num_patches_time
        encoded_grid = encoded.permute(0, 2, 1).reshape(
            batch_size, hidden_dim, num_patches_height, num_patches_time
        )
        encoded_grid = encoded_grid.permute(0, 3, 2, 1).reshape(-1, num_patches_height, hidden_dim)
        encoded_flat = encoded_grid.flatten(1)
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

    labels_downsampled = patch_labels(labels_tensor, patch_width)
    assert labels_downsampled.shape[0] == encoded_flat.shape[0]
    out = {
        "encoded_embeddings": encoded_flat.cpu().numpy().astype(np.float32, copy=False),
        "encoded_embeddings_grid": encoded_grid.cpu().numpy().astype(np.float32, copy=False),
        "labels_downsampled": labels_downsampled,
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
    segment_states = []

    for raw_segment in raw["segments"]:
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
            target_feature_type=args.get("target_feature_type", "end_of_block"),
            all_layers=bool(args.get("all_layers")),
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


def _token_count(timebins, model_num_timebins, patch_width, num_patches_time):
    batch_size = max(1, (timebins + model_num_timebins - 1) // model_num_timebins)
    pad_amount = batch_size * model_num_timebins - timebins
    return batch_size * num_patches_time - pad_amount // patch_width


def _extract_one(raw_segment, args, model_state):
    return _extract_segment_arrays(
        spec_segment=raw_segment["spectrogram"],
        labels_segment=raw_segment["labels_original"],
        model=model_state["model"],
        device=model_state["device"],
        model_num_timebins=model_state["model_num_timebins"],
        patch_width=model_state["patch_width"],
        num_patches_height=model_state["num_patches_height"],
        num_patches_time=model_state["num_patches_time"],
        encoder_layer_idx=args.get("encoder_layer_idx"),
        target_feature_type=args.get("target_feature_type", "end_of_block"),
        all_layers=bool(args.get("all_layers")),
    )


def _writer_shapes(raw_segments, first_state, model_state):
    total_tokens = sum(
        _token_count(
            segment["labels_original"].shape[0],
            model_state["model_num_timebins"],
            model_state["patch_width"],
            model_state["num_patches_time"],
        )
        for segment in raw_segments
    )
    shapes = {
        "encoded_embeddings": (total_tokens, *first_state["encoded_embeddings"].shape[1:]),
        "labels_downsampled": (total_tokens,),
        "recording_stem": (total_tokens,),
        "song_id": (total_tokens,),
        "token_start_ms": (total_tokens,),
        "token_end_ms": (total_tokens,),
    }
    if model_state.get("minimal"):
        return shapes

    total_timebins = sum(segment["labels_original"].shape[0] for segment in raw_segments)
    segment_count = len(raw_segments)
    shapes.update({
        "encoded_embeddings_grid": (total_tokens, *first_state["encoded_embeddings_grid"].shape[1:]),
        "labels_original": (total_timebins,),
        "spectrograms": (total_timebins, first_state["spectrograms"].shape[1]),
        "segment_recording_stem": (segment_count,),
        "segment_song_id": (segment_count,),
        "segment_start_ms": (segment_count,),
        "segment_end_ms": (segment_count,),
        "segment_spec_path": (segment_count,),
    })
    return shapes


def _writer_dtypes(raw_segments, first_state):
    stems = [segment["recording_stem"] for segment in raw_segments]
    stem_dtype = f"<U{max(1, max(len(stem) for stem in stems))}"
    return {
        "encoded_embeddings": first_state["encoded_embeddings"].dtype,
        "labels_downsampled": np.int64,
        "recording_stem": stem_dtype,
        "song_id": np.int64,
        "token_start_ms": np.float32,
        "token_end_ms": np.float32,
    } | ({
        "encoded_embeddings_grid": first_state["encoded_embeddings_grid"].dtype,
        "labels_original": np.int64,
        "spectrograms": np.float32,
        "segment_recording_stem": stem_dtype,
        "segment_song_id": np.int64,
        "segment_start_ms": np.float32,
        "segment_end_ms": np.float32,
        "segment_spec_path": f"<U{max(1, max(len(segment['spec_path']) for segment in raw_segments))}",
    } if not first_state.get("minimal") else {})


def _metadata(extracted, args, model_state):
    audio_params = extracted["audio_params"]
    return {
        "audio_sr": audio_params[0],
        "audio_n_mels": audio_params[1],
        "audio_hop_size": audio_params[2],
        "audio_fft": audio_params[3],
        "patch_height": model_state["patch_height"],
        "patch_width": model_state["patch_width"],
        "num_patches_time": model_state["num_patches_time"],
        "num_patches_height": model_state["num_patches_height"],
        "checkpoint": args.get("checkpoint") or "",
        "encoder_layer_idx": args.get("encoder_layer_idx"),
        "all_layers": bool(args.get("all_layers")),
        "target_feature_type": args.get("target_feature_type", "end_of_block"),
        "model_num_timebins": model_state["model_num_timebins"],
        "mels": int(model_state["config"]["mels"]),
    }


def _write_segment(writer, token_start, timebin_start, segment_index, raw_segment, state, patch_width):
    token_count = state["encoded_embeddings"].shape[0]
    timebin_count = state["labels_original"].shape[0]
    token_end = token_start + token_count
    timebin_end = timebin_start + timebin_count
    edges = np.minimum(np.arange(token_count + 1) * patch_width, timebin_count)
    edges = raw_segment["start_ms"] + edges * (
        (raw_segment["end_ms"] - raw_segment["start_ms"]) / timebin_count
    )

    writer.arrays["encoded_embeddings"][token_start:token_end] = state["encoded_embeddings"]
    writer.arrays["labels_downsampled"][token_start:token_end] = state["labels_downsampled"]
    writer.arrays["recording_stem"][token_start:token_end] = raw_segment["recording_stem"]
    writer.arrays["song_id"][token_start:token_end] = raw_segment["song_id"]
    writer.arrays["token_start_ms"][token_start:token_end] = edges[:-1]
    writer.arrays["token_end_ms"][token_start:token_end] = edges[1:]
    if "encoded_embeddings_grid" not in writer.arrays:
        return token_end, timebin_end

    writer.arrays["encoded_embeddings_grid"][token_start:token_end] = state["encoded_embeddings_grid"]
    writer.arrays["labels_original"][timebin_start:timebin_end] = state["labels_original"]
    writer.arrays["spectrograms"][timebin_start:timebin_end] = state["spectrograms"]
    writer.arrays["segment_recording_stem"][segment_index] = raw_segment["recording_stem"]
    writer.arrays["segment_song_id"][segment_index] = raw_segment["song_id"]
    writer.arrays["segment_start_ms"][segment_index] = raw_segment["start_ms"]
    writer.arrays["segment_end_ms"][segment_index] = raw_segment["end_ms"]
    writer.arrays["segment_spec_path"][segment_index] = raw_segment["spec_path"]
    return token_end, timebin_end


def write_recording_embedding_folder(args):
    model_state = load_model_state(args["run_dir"], args.get("checkpoint"), random_init=args.get("random_init", False))
    model_state["minimal"] = bool(args.get("minimal"))
    extracted = load_recording_segments(args)
    raw_segments = extracted["segments"]
    assert raw_segments, "No valid segments found for the requested recording set."

    first_state = _extract_one(raw_segments[0], args, model_state)
    assert first_state is not None, "No valid patches extracted for the requested recording set."
    first_state["minimal"] = model_state["minimal"]
    writer = EmbeddingFolderWriter(
        args["out_dir"],
        _writer_shapes(raw_segments, first_state, model_state),
        _writer_dtypes(raw_segments, first_state),
        _metadata(extracted, args, model_state),
    )

    patch_width = model_state["patch_width"]
    token_start, timebin_start = _write_segment(writer, 0, 0, 0, raw_segments[0], first_state, patch_width)
    for segment_index, raw_segment in enumerate(raw_segments[1:], start=1):
        state = _extract_one(raw_segment, args, model_state)
        assert state is not None, "No valid patches extracted for a non-empty segment."
        token_start, timebin_start = _write_segment(
            writer, token_start, timebin_start, segment_index, raw_segment, state, patch_width
        )

    assert token_start == writer.arrays["encoded_embeddings"].shape[0]
    if "spectrograms" in writer.arrays:
        assert timebin_start == writer.arrays["spectrograms"].shape[0]
    writer.close()


def main(args):
    if args.get("out_dir"):
        write_recording_embedding_folder(args)
        return None
    return extract_recording_embeddings(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SongMAE embeddings for exact recordings or full directories.")
    parser.add_argument("--num_timebins", type=int, default=12400)
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--spec_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--bird", type=str, default=None)
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--all_layers", action="store_true")
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
    parser.add_argument(
        "--target_feature_type",
        default="end_of_block",
        choices=TARGET_FEATURE_TYPES,
    )
    args = parser.parse_args()
    main(vars(args))
