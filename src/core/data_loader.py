from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import numpy as np
from .utils import create_label_arr, load_json_events, load_spec, normalize_spectrogram
from .data_structures import AudioParams

""" Goal is to stay ~100 LoC here """

# loading files
class SpectrogramDataset(Dataset):
    def __init__(self, dir, n_timebins=1024, normalize=True):
        """
        n_timebins = None means no cropping
        """
        self.file_dirs = sorted(list(Path(dir).glob("*.npy")))
        if not len(self.file_dirs): raise SystemExit("no files!")

        self.params = AudioParams.from_dir(dir)
        self.mean = np.float32(self.params.mean)
        self.std = np.float32(self.params.std)
        self.n_timebins = n_timebins
        self.normalize = normalize

    def _crop_pad(self, arr):
        """
        Crop to a random n_timebins window, or zero-pad if too short.
        Returns (arr, start, end) where [start:end] is the kept source window.
        """
        t = arr.shape[1]

        # whole file
        if self.n_timebins is None:
            return arr[:, :], 0, t

        # crop
        if t > self.n_timebins:
            start = random.randint(0, t - self.n_timebins)
            end = start + self.n_timebins
            return arr[:, start:end], start, end

        # exact
        if t == self.n_timebins:
            return arr[:, :], 0, t

        # pad
        pad_amount = self.n_timebins - t
        arr = np.pad(arr[:, :], ((0, 0), (0, pad_amount)), mode="constant")
        return arr, 0, t

    def __getitem__(self, index):
        path = self.file_dirs[index]
        fname = path.stem
        arr = load_spec(path)

        arr, _, _ = self._crop_pad(arr)

        if self.normalize:
            arr = normalize_spectrogram(arr, self.mean, self.std)
        spec = torch.from_numpy(arr).unsqueeze(0)

        return spec, fname

    def __len__(self):
        return len(self.file_dirs)


class SpectrogramDatasetSupervised(SpectrogramDataset):
    def __init__(
        self,
        dir,
        annotation_file=None,
        n_timebins=1024,
        recording_mode="full_recordings",
        recording_stem=None,
        recording_stems=None,
        selected_bird=None,
        normalize=True,
    ):
        super().__init__(dir, n_timebins=n_timebins, normalize=normalize)
        self.recording_mode = recording_mode
        if recording_stems is not None:
            wanted = set(recording_stems)
            self.file_dirs = [path for path in self.file_dirs if path.stem in wanted]
        if recording_stem is not None:
            self.file_dirs = [path for path in self.file_dirs if path.stem == recording_stem]

        audio_params = (self.params.sr, self.params.mels, self.params.hop_size, self.params.fft)
        self.events_by_file = (
            load_json_events(annotation_file, audio_params, selected_bird=selected_bird)
            if annotation_file
            else {path.stem: [] for path in self.file_dirs}
        )
        self.file_dirs = [path for path in self.file_dirs if path.stem in self.events_by_file]
        if not len(self.file_dirs): raise SystemExit("no labeled files!")
        self.samples = self._build_samples()

    def _build_samples(self):
        # full_recordings samples regular file crops; events samples one crop per detected event.
        if self.recording_mode == "full_recordings":
            return [(path, None) for path in self.file_dirs]
        if self.recording_mode == "events":
            return [
                (path, event)
                for path in self.file_dirs
                for event in self.events_by_file[path.stem]
            ]
        raise ValueError(f"unknown recording mode: {self.recording_mode}")

    def _event_crop(self, arr, event):
        # Anchor fixed-size supervised windows around the event so labels are not mostly background.
        if self.n_timebins is None:
            start = max(0, int(event["on_timebins"]))
            end = min(arr.shape[1], int(event["off_timebins"]))
            return arr[:, start:end], start, end

        t = arr.shape[1]
        if t <= self.n_timebins:
            return self._crop_pad(arr)

        event_start = max(0, min(int(event["on_timebins"]), t - 1))
        event_end = max(event_start + 1, min(int(event["off_timebins"]), t))
        lo = max(0, event_end - self.n_timebins)
        hi = min(event_start, t - self.n_timebins)
        start = random.randint(lo, hi) if lo <= hi else max(0, min(event_start, t - self.n_timebins))
        end = start + self.n_timebins
        return arr[:, start:end], start, end

    def __getitem__(self, index):
        path, event = self.samples[index]
        arr = load_spec(path)
        arr, start, _ = self._event_crop(arr, event) if event else self._crop_pad(arr)
        events = self.events_by_file[path.stem]
        units = [unit for event in events for unit in event["units"]]
        labels = create_label_arr({"units": units}, start, start + arr.shape[1])

        if self.normalize:
            arr = normalize_spectrogram(arr, self.mean, self.std)
        spec = torch.from_numpy(arr).unsqueeze(0)
        labels = torch.from_numpy(labels)
        return spec, labels, path.stem

    def __len__(self):
        return len(self.samples)
