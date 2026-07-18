from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.data_loader import SpectrogramDatasetSupervised
from src.core.embedding_store import save_embedding_arrays
from src.core.utils import downsample_labels, timebins_to_ms


class WavFromSpectrogramDataset(Dataset):
    def __init__(
        self,
        spec_dir,
        wav_dir,
        annotation_file,
        recording_mode="events",
        recording_stem=None,
        recording_stems=None,
        selected_bird=None,
        wav_exts=(".wav", ".flac", ".ogg", ".mp3"),
    ):
        self.spec_dataset = SpectrogramDatasetSupervised(
            spec_dir,
            annotation_file,
            n_timebins=None,
            recording_mode=recording_mode,
            recording_stem=recording_stem,
            recording_stems=recording_stems,
            selected_bird=selected_bird,
            normalize=False,
        )
        self.wav_paths = self._index_wavs(wav_dir, wav_exts)
        self.audio_params = (
            self.spec_dataset.params.sr,
            self.spec_dataset.params.mels,
            self.spec_dataset.params.hop_size,
            self.spec_dataset.params.fft,
        )

    def _wav_exts(self, wav_exts):
        if isinstance(wav_exts, str):
            wav_exts = wav_exts.split(",")
        return {ext.strip().lower() for ext in wav_exts if ext.strip()}

    def _index_wavs(self, wav_dir, wav_exts):
        exts = self._wav_exts(wav_exts)
        paths = {}
        for path in Path(wav_dir).rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                assert path.stem not in paths, f"duplicate wav stem: {path.stem}"
                paths[path.stem] = path
        assert paths, f"no wav files found: {wav_dir}"
        return paths

    def __getitem__(self, index):
        # Wrap the SongMAE spectrogram loader so raw-audio models use the same
        # exact files, recording filters, event windows, and JSON labels.
        _, labels, stem = self.spec_dataset[index]
        spec_path, event = self.spec_dataset.samples[index]
        wav_stem = spec_path.with_suffix("").name
        assert wav_stem in self.wav_paths, f"missing wav for spec: {spec_path}"
        start = 0 if event is None else max(0, int(event["on_timebins"]))
        end = start + int(labels.numel())
        return {
            "spec_path": spec_path,
            "wav_path": self.wav_paths[wav_stem],
            "recording_stem": stem,
            "song_id": index,
            "start_ms": timebins_to_ms(start, self.audio_params),
            "end_ms": timebins_to_ms(end, self.audio_params),
            "labels": labels,
        }

    def __len__(self):
        return len(self.spec_dataset)


def limited_items(dataset, num_timebins, indices=None):
    max_timebins = int(num_timebins or 0)
    used = 0
    for index in indices if indices is not None else range(len(dataset)):
        item = dataset[index]
        labels = item["labels"]
        count = int(labels.numel())
        if max_timebins > 0:
            remaining = max_timebins - used
            if remaining <= 0:
                return
            if count > remaining:
                item = dict(item)
                item["labels"] = labels[:remaining]
                item["end_ms"] = item["start_ms"] + timebins_to_ms(remaining, dataset.audio_params)
                count = remaining
        yield item
        used += count


def chunked_items(dataset, num_timebins, chunk_timebins, indices=None):
    size = int(chunk_timebins or 0)
    for item in limited_items(dataset, num_timebins, indices):
        labels = item["labels"]
        count = int(labels.numel())
        if size <= 0:
            yield item
            continue
        for start in range(0, count, size):
            chunk = dict(item)
            chunk["labels"] = labels[start : start + size]
            chunk["start_ms"] = item["start_ms"] + timebins_to_ms(start, dataset.audio_params)
            chunk["end_ms"] = item["start_ms"] + timebins_to_ms(min(start + size, count), dataset.audio_params)
            yield chunk


def append_limited(rows, row, max_points, used):
    count = int(row["encoded_embeddings"].shape[0])
    max_points = int(max_points or 0)
    if max_points <= 0:
        rows.append(row)
        return used + count, True

    remaining = max_points - used
    if remaining <= 0:
        return used, False

    if count > remaining:
        row = dict(row)
        item = dict(row["item"])
        if "token_edges" in row:
            row["token_edges"] = row["token_edges"][: remaining + 1]
        else:
            item["end_ms"] = item["start_ms"] + (item["end_ms"] - item["start_ms"]) * remaining / count
        row["item"] = item
        row["encoded_embeddings"] = row["encoded_embeddings"][:remaining]
        row["labels_downsampled"] = row["labels_downsampled"][:remaining]
        if "encoded_embeddings_grid" in row:
            row["encoded_embeddings_grid"] = row["encoded_embeddings_grid"][:remaining]
        count = remaining

    rows.append(row)
    used += count
    return used, used < max_points


def convolution_geometry(kernels, strides, samples_per_timebin):
    assert len(kernels) == len(strides) and samples_per_timebin > 0
    receptive_field = 1
    step = 1
    for kernel, stride in zip(kernels, strides):
        receptive_field += (int(kernel) - 1) * step
        step *= int(stride)
    return receptive_field / (2 * samples_per_timebin), step / samples_per_timebin


def convolution_feature_map(labels, output_length, geometry):
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.ndim == 1 and labels.size > 0 and output_length > 0

    first_center, stride = geometry
    centers = first_center + np.arange(output_length) * stride
    indices = np.minimum(centers.astype(np.int64), labels.size - 1)
    edges = np.empty(output_length + 1, dtype=np.float64)
    edges[0], edges[-1] = 0, labels.size
    if output_length > 1:
        edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    assert np.all(edges[1:] > edges[:-1])
    return labels[indices], edges


def save_concatenated_embeddings(out_dir, rows, **metadata):
    assert rows
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, original_labels, stems, song_ids, starts, ends = [], [], [], [], [], [], []
    segment_stems, segment_song_ids, segment_starts, segment_ends = [], [], [], []
    spec_paths, wav_paths = [], []
    grids = []

    for row in rows:
        item = row["item"]
        x = row["encoded_embeddings"]
        y = row["labels_downsampled"]
        assert x.shape[0] == y.shape[0]
        count = x.shape[0]
        label_count = int(item["labels"].numel())
        source_edges = row.get("token_edges")
        if source_edges is not None:
            source_edges = np.asarray(source_edges)
            assert source_edges.shape == (count + 1,)
        elif label_count >= count:
            source_edges = np.rint(np.linspace(0, label_count, count + 1)).astype(np.int64)
        else:
            source_edges = np.linspace(0, label_count, count + 1)
        edges = item["start_ms"] + source_edges * (
            (item["end_ms"] - item["start_ms"]) / label_count
        )

        features.append(x.astype(np.float32, copy=False))
        labels.append(y.astype(np.int64, copy=False))
        original_labels.append(item["labels"].numpy().astype(np.int64, copy=False))
        stems.append(np.full(count, item["recording_stem"]))
        song_ids.append(np.full(count, item["song_id"], dtype=np.int64))
        starts.append(edges[:-1].astype(np.float32, copy=False))
        ends.append(edges[1:].astype(np.float32, copy=False))

        segment_stems.append(item["recording_stem"])
        segment_song_ids.append(item["song_id"])
        segment_starts.append(item["start_ms"])
        segment_ends.append(item["end_ms"])
        spec_paths.append(str(item["spec_path"]))
        wav_paths.append(str(item["wav_path"]))
        if "encoded_embeddings_grid" in row:
            grids.append(row["encoded_embeddings_grid"].astype(np.float32, copy=False))

    arrays = {
        "encoded_embeddings": np.concatenate(features, axis=0),
        "labels_downsampled": np.concatenate(labels, axis=0),
        "labels_original": np.concatenate(original_labels, axis=0),
        "recording_stem": np.concatenate(stems, axis=0),
        "song_id": np.concatenate(song_ids, axis=0),
        "token_start_ms": np.concatenate(starts, axis=0),
        "token_end_ms": np.concatenate(ends, axis=0),
        "segment_recording_stem": np.asarray(segment_stems),
        "segment_song_id": np.asarray(segment_song_ids, dtype=np.int64),
        "segment_start_ms": np.asarray(segment_starts, dtype=np.float32),
        "segment_end_ms": np.asarray(segment_ends, dtype=np.float32),
        "segment_spec_path": np.asarray(spec_paths),
        "segment_wav_path": np.asarray(wav_paths),
    }
    if grids:
        assert len(grids) == len(rows)
        arrays["encoded_embeddings_grid"] = np.concatenate(grids, axis=0)
    save_embedding_arrays(out_dir, arrays, metadata)


def labels_for_features(labels, output_length):
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.ndim == 1
    assert output_length > 0
    assert labels.size > 0
    if labels.size >= output_length:
        return downsample_labels(labels, output_length)

    index = np.floor((np.arange(output_length) + 0.5) * labels.size / output_length).astype(np.int64)
    index = np.minimum(index, labels.size - 1)
    return labels[index].astype(np.int64, copy=False)
