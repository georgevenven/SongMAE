import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from data_loader import SpectrogramDatasetSupervised
from utils import normalize_spectrogram
from utils import load_audio_params, load_model_state

def fit_feature_postprocess(features, mode="none", dim=256):
    if mode == "none":
        return None

    if mode == "whiten_l2":
        std = features.std(axis=0)
        std = np.maximum(std, 1e-12)
        return {
            "kind": mode,
            "dim": int(features.shape[1]),
            "mean": features.mean(axis=0).astype(np.float32, copy=False),
            "std": std.astype(np.float32, copy=False),
        }

    n_components = min(int(dim), int(features.shape[0]), int(features.shape[1]))
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
        payload["std"] = transform["std"]
    np.savez(path, **payload)


def load_feature_postprocess(path):
    loaded = np.load(path)
    kind = str(loaded["kind"].item())
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
        whitened = (centered / np.maximum(transform["std"], 1e-12)).astype(np.float32, copy=False)
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (whitened / norms).astype(np.float32, copy=False)


# Fit or load one whitening/PCA transform, apply it to one feature matrix, and optionally save it.
def maybe_apply_feature_postprocess(
    features,
    mode="none",
    dim=256,
    load_path=None,
    save_path=None,
):
    if mode == "none":
        return features.astype(np.float32, copy=False), None

    if load_path is not None:
        transform = load_feature_postprocess(load_path)
    else:
        transform = fit_feature_postprocess(features, mode=mode, dim=dim)
        if save_path is not None:
            save_feature_postprocess(save_path, transform)
    return apply_feature_postprocess_transform(features, transform), transform


# Fit one transform across all segment features, then write transformed features back per segment.
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

    load_path = args.get("embedding_postprocess_load")
    if load_path is not None:
        transform = load_feature_postprocess(load_path)
        for segment in segments:
            if segment[feature_key].shape[0] > 0:
                segment[feature_key] = apply_feature_postprocess_transform(segment[feature_key], transform)
        return {
            "mode": mode,
            "dim": int(transform["dim"]),
            "feature_key": feature_key,
            "load_path": load_path,
            "save_path": args.get("embedding_postprocess_save"),
        }

    stacked = np.concatenate(arrays, axis=0)
    transformed, transform = maybe_apply_feature_postprocess(
        stacked,
        mode=mode,
        dim=args.get("embedding_postprocess_dim", 256),
        load_path=None,
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
    for idx in range(len(ds)):
        if max_timebins is not None and collected_timebins >= max_timebins:
            break
        spec, labels, stem = ds[idx]
        spec = spec.squeeze(0).numpy()
        labels = labels.numpy()
        if max_timebins is not None:
            remaining = max_timebins - collected_timebins
            spec = spec[:, :remaining]
            labels = labels[:remaining]
        if spec.shape[1] == 0:
            continue
        segments.append({"recording_stem": stem, "spectrogram": spec, "labels_original": labels})
        collected_timebins += spec.shape[1]

    return {
        "audio_params": (ds.params.sr, ds.params.mels, ds.params.hop_size, ds.params.fft),
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
    model_state = load_model_state(args["run_dir"], args.get("checkpoint"))
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
