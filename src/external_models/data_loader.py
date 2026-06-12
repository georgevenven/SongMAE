from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.data_loader import SpectrogramDatasetSupervised
from src.core.utils import timebins_to_ms


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
        start = 0 if event is None else int(event["on_timebins"])
        end = int(labels.numel()) if event is None else int(event["off_timebins"])
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


def save_concatenated_embeddings(out_dir, rows, **metadata):
    assert rows
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, stems, song_ids, starts, ends = [], [], [], [], [], []
    segment_stems, segment_song_ids, segment_starts, segment_ends = [], [], [], []
    spec_paths, wav_paths = [], []
    grids = []

    for row in rows:
        item = row["item"]
        x = row["encoded_embeddings"]
        y = row["labels_downsampled"]
        assert x.shape[0] == y.shape[0]
        count = x.shape[0]
        edges = np.linspace(item["start_ms"], item["end_ms"], count + 1)

        features.append(x.astype(np.float32, copy=False))
        labels.append(y.astype(np.int64, copy=False))
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

    payload = {
        "encoded_embeddings": np.concatenate(features, axis=0),
        "labels_downsampled": np.concatenate(labels, axis=0),
        "labels_original": np.concatenate(labels, axis=0),
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
        payload["encoded_embeddings_grid"] = np.concatenate(grids, axis=0)
    for key, value in metadata.items():
        payload[key] = np.asarray(value)

    tmp_path = out_dir / "embeddings.tmp.npz"
    out_path = out_dir / "embeddings.npz"
    np.savez(tmp_path, **payload)
    tmp_path.replace(out_path)
    print(f"NPZ saved to {out_path}")
