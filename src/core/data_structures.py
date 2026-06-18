"""Canonical JSON shapes for SongMAE data, models, labels, and training."""

# File locations:
# - audio_params.json lives beside each spectrogram dataset, and is copied into runs.
# - model.json lives in the run folder.
# - train.json lives in the run folder.
# - labels live in files/, beside the corresponding spectrogram dataset, and in
#   supervised model run folders.

import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path


def _read_json(path):
    path = Path(path)
    assert path.exists(), f"missing JSON file: {path}"
    return json.loads(path.read_text())


def _load_dataclass(cls, path):
    data = _read_json(path)
    model_fields = [field for field in fields(cls) if field.init]
    required = [
        field.name
        for field in model_fields
        if field.default is MISSING and field.default_factory is MISSING
    ]
    missing = [name for name in required if name not in data]
    assert not missing, f"{Path(path).name} missing keys: {missing}"

    kwargs = {field.name: data[field.name] for field in model_fields if field.name in data}
    return cls(**kwargs)


class JsonDataclass:
    @classmethod
    def from_json(cls, path):
        return _load_dataclass(cls, path)


@dataclass(frozen=True)
class AudioParams(JsonDataclass):
    mels: int
    sr: int
    hop_size: int
    fft: int
    mean: float
    std: float

    @classmethod
    def from_dir(cls, data_dir):
        return cls.from_json(Path(data_dir) / "audio_params.json")


@dataclass(frozen=True)
class ModelConfig(JsonDataclass):
    mels: int
    num_timebins: int
    patch_height: int
    patch_width: int
    enc_hidden_d: int
    enc_n_head: int
    enc_n_layer: int
    enc_dim_ff: int
    dec_hidden_d: int
    dec_n_head: int
    dec_n_layer: int
    dec_dim_ff: int
    dropout: float
    mask_p: float
    mask_c: float
    mask_type: str = "voronoi"

    def __post_init__(self):
        assert self.mels % self.patch_height == 0
        assert self.num_timebins % self.patch_width == 0

    @property
    def patch_size(self):
        return (self.patch_height, self.patch_width)

    @property
    def max_seq(self):
        return self.num_patches_height * self.num_patches_time

    @property
    def num_patches_height(self):
        return self.mels // self.patch_height

    @property
    def num_patches_time(self):
        return self.num_timebins // self.patch_width


@dataclass(frozen=True)
class TrainConfig(JsonDataclass):
    train_dir: str
    val_dir: str
    run_name: str
    steps: int
    batch_size: int
    lr: float
    task: str = "unsupervised"
    weight_decay: float = 0.0
    num_workers: int = 4
    eval_every: int = 500
    warmup_steps: int = 0
    min_lr: float = 0.0
    amp: bool = False
    amp_dtype: str = "bf16"
    wandb: bool = False
    annotation_file: str | None = None
    recording_mode: str = "full_recordings"

    def __post_init__(self):
        assert self.task in ("unsupervised", "supervised")
        assert self.recording_mode in ("events", "full_recordings")
        assert self.amp_dtype in ("bf16", "fp16")
        assert self.task == "unsupervised" or self.annotation_file


@dataclass(frozen=True)
class Labels:
    recordings: list[dict]
    metadata: dict | None = None

    @classmethod
    def from_json(cls, path):
        data = _read_json(path)
        assert "recordings" in data, f"{Path(path).name} missing keys: ['recordings']"
        return cls(recordings=data["recordings"], metadata=data.get("metadata", {}))

    def for_file(self, filename):
        stem = Path(filename).stem
        for recording in self.recordings:
            rec_stem = Path(recording["recording"]["filename"]).stem
            if rec_stem == stem:
                return recording
        raise KeyError(f"no labels for file: {filename}")

    def events(self, filename):
        return self.for_file(filename).get("detected_events", [])

    def class_id_map(self):
        ids = set()
        for recording in self.recordings:
            for event in recording.get("detected_events", []):
                for unit in event.get("units", []):
                    ids.add(int(unit["id"]))
        return {raw_id: idx for idx, raw_id in enumerate(sorted(ids))}

    def unit_label_map(self):
        return (self.metadata or {}).get("unit_id_to_label", {})

    def num_classes(self, mode):
        assert mode in ("detect", "unit_detect", "classify")
        if mode in ("detect", "unit_detect"):
            return 2
        return len(self.class_id_map()) + 1
