from pathlib import Path
import sys

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
            "start_ms": timebins_to_ms(start, self.audio_params),
            "end_ms": timebins_to_ms(end, self.audio_params),
            "labels": labels,
        }

    def __len__(self):
        return len(self.spec_dataset)
