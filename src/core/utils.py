"""Small shared helpers for the refactored SongMAE code."""

import json
import re
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"
CHUNK_MS_RE = re.compile(r"^(?P<base>.+)__ms_(?P<start>\d+)_(?P<end>\d+)$")

def write_spec(path, spec, storage_dtype="float32"):
    assert storage_dtype in ("float32", "int8_affine")
    path = Path(path)
    spec = np.asarray(spec, dtype=np.float32)
    if storage_dtype == "float32":
        np.save(path, spec.T)
        return path

    low = spec.min(axis=1)
    high = spec.max(axis=1)
    scale = np.maximum((high - low) / 255.0, np.float32(1e-6))
    offset = low + 128.0 * scale
    codes = np.clip(np.rint((spec - offset[:, None]) / scale[:, None]), -128, 127)
    np.save(path, codes.T.astype(np.int8))
    np.savetxt(path.with_suffix(".txt"), np.column_stack([scale, offset]), fmt="%.9g")
    return path


def dequant_int8(codes, affine):
    spec = codes.astype(np.float32) * affine[None, :, 0] + affine[None, :, 1]
    return np.array(spec.T, dtype=np.float32, copy=True)


def load_int8_spec(path):
    codes = np.load(path, mmap_mode="r")
    affine = np.loadtxt(Path(path).with_suffix(".txt"), dtype=np.float32)
    affine = np.atleast_2d(affine)
    return dequant_int8(codes, affine)


def load_spec(path):
    arr = np.load(path, mmap_mode="r")
    if arr.dtype == np.int8:
        return load_int8_spec(path)
    return np.array(arr.T, dtype=np.float32, copy=True)


def spec_timebins(path):
    arr = np.load(path, mmap_mode="r")
    return int(arr.shape[0])


def load_spec_slice(path, start, end):
    arr = np.load(path, mmap_mode="r")
    sliced = arr[start:end]
    if arr.dtype == np.int8:
        affine = np.loadtxt(Path(path).with_suffix(".txt"), dtype=np.float32)
        affine = np.atleast_2d(affine)
        return dequant_int8(sliced, affine)
    return np.array(sliced.T, dtype=np.float32, copy=True)


def list_spec_items(data_dir):
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.npy"))
    if paths:
        return paths

    index = data_dir / "shards" / "index.tsv"
    if not index.exists():
        return []

    items = []
    for line in index.read_text().splitlines()[1:]:
        name, shard, start, end = line.split("\t")
        items.append((data_dir / "shards" / shard, int(start), int(end), name))
    return items


def load_spec_item(item):
    if isinstance(item, Path):
        return load_spec(item), item.stem
    path, start, end, name = item
    return load_spec_slice(path, start, end), name

def normalize_spectrogram(arr, mean, std):
    arr = np.asarray(arr, dtype=np.float32)
    return ((arr - np.float32(mean)) / np.float32(std)).astype(np.float32)


def normalize_spectrogram_tensor(arr, mean, std):
    mean = torch.as_tensor(mean, dtype=arr.dtype, device=arr.device)
    std = torch.as_tensor(std, dtype=arr.dtype, device=arr.device)
    return (arr - mean) / std


def load_audio_params(data_dir, require_stats=True):
    path = Path(data_dir) / "audio_params.json"
    return json.loads(path.read_text())


def ms_to_timebins(ms_value, audio_params):
    sr = audio_params[0]
    hop_size = audio_params[2]
    return int((ms_value / 1000) * sr / hop_size)


def timebins_to_ms(timebin_value, audio_params):
    sr = audio_params[0]
    hop_size = audio_params[2]
    return float(timebin_value) * hop_size / sr * 1000


def majority_label(labels):
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.ndim == 1 and labels.size
    values, counts = np.unique(labels, return_counts=True)
    winners = set(values[counts == counts.max()].tolist())
    center = int(labels[labels.size // 2])
    if center in winners:
        return center
    return next(int(label) for label in labels if int(label) in winners)


def labels_by_edges(labels, edges):
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    assert labels.ndim == 1 and edges.ndim == 1
    assert edges[0] == 0 and edges[-1] == labels.size
    assert np.all(edges[1:] > edges[:-1])
    return np.asarray([majority_label(labels[start:end]) for start, end in zip(edges[:-1], edges[1:])])


def patch_labels(labels, width):
    count = (len(labels) + width - 1) // width
    edges = np.minimum(np.arange(count + 1) * width, len(labels))
    return labels_by_edges(labels, edges)


def load_json_events(json_path, audio_params, selected_bird=None):
    data = json.loads(Path(json_path).read_text())
    event_map = {}
    for rec in data.get("recordings", []):
        recording = rec.get("recording", {})
        bird_id = recording.get("bird_id", "")
        stem = Path(recording.get("filename", "")).stem
        if selected_bird is not None and bird_id != selected_bird:
            continue

        events = []
        for event in rec.get("detected_events", []):
            units = [
                (
                    ms_to_timebins(unit["onset_ms"], audio_params),
                    ms_to_timebins(unit["offset_ms"], audio_params),
                    int(unit["id"]),
                )
                for unit in event.get("units", [])
            ]
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


def downsample_labels(labels, output_length):
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.ndim == 1
    assert output_length > 0
    assert labels.size >= output_length
    edges = np.rint(np.linspace(0, labels.size, output_length + 1)).astype(np.int64)
    return labels_by_edges(labels, edges)


def resolve_run_dir(run_dir):
    path = Path(run_dir)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return RUNS_ROOT / path


def load_model_from_checkpoint(run_dir, checkpoint_file=None, fallback_to_random=False):
    from .model import SongMAE

    run_dir = resolve_run_dir(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    model = SongMAE(config)
    if fallback_to_random:
        print(f"random initialized model from config: {run_dir}")
        return model, config

    if checkpoint_file is None:
        checkpoint = sorted((run_dir / "weights").glob("model_step_*.pth"))[-1]
    else:
        checkpoint = Path(checkpoint_file)
        if not checkpoint.is_absolute():
            checkpoint = run_dir / "weights" / checkpoint

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    print(f"loaded checkpoint: {checkpoint}")
    return model, config


def load_model_state(run_dir, checkpoint_file=None, random_init=False):
    from .data_structures import ModelConfig

    run_dir = resolve_run_dir(run_dir)
    model, config = load_model_from_checkpoint(run_dir, checkpoint_file, fallback_to_random=random_init)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    model_json = run_dir / "model.json"
    model_config = ModelConfig.from_json(model_json if model_json.exists() else run_dir / "config.json")
    return {
        "model": model,
        "config": config,
        "model_config": model_config,
        "run_dir": str(run_dir),
        "device": device,
        "patch_height": model_config.patch_height,
        "patch_width": model_config.patch_width,
        "model_num_timebins": model_config.num_timebins,
        "num_patches_time": model_config.num_patches_time,
        "num_patches_height": model_config.num_patches_height,
    }


def resolve_single_spec_path(spec_dir, recording_stem):
    spec_dir = Path(spec_dir)
    path = spec_dir / f"{recording_stem}.npy"
    if path.exists():
        return path

    paths = sorted(spec_dir.rglob(f"{recording_stem}.npy"))
    assert paths, f"Recording not found: {recording_stem}"
    assert len(paths) == 1, f"Multiple recordings found: {recording_stem}"
    return paths[0]


def parse_chunk_ms(filename):
    name = Path(filename).name
    for suffix in (".npy", ".wav"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    start = end = None
    while True:
        match = CHUNK_MS_RE.match(name)
        if match is None:
            return name, start, end
        name = match.group("base")
        start = int(match.group("start"))
        end = int(match.group("end"))


def clip_labels_to_chunk(labels, chunk_start_ms, chunk_end_ms):
    if chunk_start_ms is None:
        return labels
    chunk_end_ms = chunk_end_ms if chunk_end_ms is not None else float("inf")
    clipped = []
    for label in labels:
        onset = label["onset_ms"]
        offset = label["offset_ms"]
        if offset <= chunk_start_ms or onset >= chunk_end_ms:
            continue
        label = dict(label)
        label["onset_ms"] = max(onset, chunk_start_ms) - chunk_start_ms
        label["offset_ms"] = min(offset, chunk_end_ms) - chunk_start_ms
        clipped.append(label)
    return clipped


def get_class_id_map_from_annotations(path, mode):
    if mode != "classify":
        return {}
    annotations = json.loads(Path(path).read_text())
    ids = {
        int(unit["id"])
        for recording in annotations["recordings"]
        for event in recording.get("detected_events", [])
        for unit in event.get("units", [])
    }
    return {raw_id: idx for idx, raw_id in enumerate(sorted(ids))}


def get_num_classes_from_annotations(path, mode):
    if mode in ("detect", "unit_detect"):
        return 2
    return len(get_class_id_map_from_annotations(path, mode)) + 1
