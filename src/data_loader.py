## LATER TO DO:
### THESE TWO CLASSES SHOULD JUST INHERIT A MORE BASIC CLASS
##

import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import numpy as np
from utils import (
    load_audio_params,
    parse_chunk_ms,
    clip_labels_to_chunk,
    get_class_id_map_from_annotations,
    get_num_classes_from_annotations,
)

SUPPORTED_NORMALIZATIONS = {"none", "audio_params", "per_file_zscore"}
SUPPORTED_OUTPUT_DTYPES = {"float32", "float16", "bfloat16"}
SUPPORTED_STORAGE_DTYPES = {"float32", "float8_e4m3fn", "float8_e5m2"}
SUPPORTED_STORAGE_NORMALIZATIONS = {"none", "per_file_zscore"}


def resolve_output_dtype(output_dtype):
    assert output_dtype in SUPPORTED_OUTPUT_DTYPES
    if output_dtype == "float32":
        return torch.float32
    if output_dtype == "float16":
        return torch.float16
    return torch.bfloat16


def load_quantization_manifest(data_dir, audio_data_json):
    storage_normalization = audio_data_json.get("storage_normalization", "none")
    manifest_name = audio_data_json.get("storage_manifest")
    if storage_normalization != "per_file_zscore":
        return {}
    assert manifest_name is not None
    manifest_path = Path(data_dir) / manifest_name
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["file"]: row for row in rows}


def decode_fp8_array(arr, storage_dtype):
    assert storage_dtype in {"float8_e4m3fn", "float8_e5m2"}
    tensor = torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True))
    if storage_dtype == "float8_e4m3fn":
        return tensor.view(torch.float8_e4m3fn).float().numpy()
    return tensor.view(torch.float8_e5m2).float().numpy()


def decode_storage_to_raw(arr, storage_dtype, storage_normalization, file_stats):
    assert storage_dtype in SUPPORTED_STORAGE_DTYPES
    assert storage_normalization in SUPPORTED_STORAGE_NORMALIZATIONS

    if storage_dtype == "float32":
        stored = np.asarray(arr, dtype=np.float32)
    else:
        stored = decode_fp8_array(arr, storage_dtype)

    if storage_normalization == "none":
        return stored.astype(np.float32, copy=False)

    assert file_stats is not None
    mean = np.float32(file_stats["mean"])
    std = np.float32(file_stats["std"])
    return (stored * std + mean).astype(np.float32, copy=False)


def normalize_spectrogram_numpy(arr, normalization, mean=None, std=None):
    arr = np.asarray(arr, dtype=np.float32)
    assert normalization in SUPPORTED_NORMALIZATIONS
    if normalization == "none":
        return arr.astype(np.float32, copy=False)
    if normalization == "audio_params":
        assert mean is not None
        assert std is not None
        return ((arr - np.float32(mean)) / np.float32(std)).astype(np.float32, copy=False)

    arr_mean = np.float32(arr.mean())
    arr_std = max(np.float32(arr.std()), np.float32(1e-6))
    return ((arr - arr_mean) / arr_std).astype(np.float32, copy=False)


def normalize_spectrogram_tensor(arr, normalization, mean=None, std=None, dims=None):
    assert normalization in SUPPORTED_NORMALIZATIONS
    if dims is None:
        dims = tuple(range(1, arr.ndim))

    if normalization == "none":
        return arr
    if normalization == "audio_params":
        assert mean is not None
        assert std is not None
        mean_tensor = torch.as_tensor(mean, dtype=arr.dtype, device=arr.device)
        std_tensor = torch.as_tensor(std, dtype=arr.dtype, device=arr.device)
        return (arr - mean_tensor) / std_tensor

    arr_mean = arr.mean(dim=dims, keepdim=True)
    arr_std = arr.std(dim=dims, keepdim=True)
    arr_std = torch.clamp(arr_std, min=1e-6)
    return (arr - arr_mean) / arr_std

class SpectogramDataset(Dataset):
    def __init__(self, dir, n_timebins=1024, normalization="audio_params", output_dtype="float32"):
        """
        n_timebins = None means no cropping
        """
        self.file_dirs = sorted(list(Path(dir).glob("*.npy")))

        # Load audio parameters using utility function
        self.audio_data_json = load_audio_params(dir)
        
        self.n_mels = self.audio_data_json["mels"]
        self.sr = self.audio_data_json["sr"]
        self.hop_size = self.audio_data_json["hop_size"]
        self.fft = self.audio_data_json["fft"]
        self.mean = self.audio_data_json["mean"]
        self.std = self.audio_data_json["std"]
        self.n_timebins = n_timebins
        self.normalization = normalization
        self.output_dtype = output_dtype
        self.mean = np.float32(self.mean)
        self.std = np.float32(self.std)
        assert self.normalization in SUPPORTED_NORMALIZATIONS
        self.output_torch_dtype = resolve_output_dtype(output_dtype)
        self.storage_dtype = self.audio_data_json.get("storage_dtype", "float32")
        self.storage_normalization = self.audio_data_json.get("storage_normalization", "none")
        assert self.storage_dtype in SUPPORTED_STORAGE_DTYPES
        assert self.storage_normalization in SUPPORTED_STORAGE_NORMALIZATIONS
        self.quantization_manifest = load_quantization_manifest(dir, self.audio_data_json)

        if len(self.file_dirs) == 0:
            raise SystemExit("no files!!")
            
    def __getitem__(self, index):
        path = self.file_dirs[index]  

        # if file not possible to open pick a random file from the list and try again, this actually shoudl recursively recall getitem 
        filename = path.stem
        arr = np.load(path, mmap_mode="r")
        time = arr.shape[1]
        file_stats = self.quantization_manifest.get(path.name)

        # we want to load the whole file if n_timebins is None
        if self.n_timebins is None:
            start = 0
            end = time
            arr = arr[:, start:end]

        # loading a chunk 
        else:
            # crop 
            if time > self.n_timebins:
                start = random.randint(0, time - self.n_timebins)
                end = start + self.n_timebins
                arr = arr[:, start:end]

            # do nothing 
            if time == self.n_timebins:
                start = 0
                end = time
                arr = arr[:, start:end]

            # pad 
            if time < self.n_timebins:
                start = 0
                end = self.n_timebins
                arr = arr[:, :]

        arr = decode_storage_to_raw(
            arr,
            self.storage_dtype,
            self.storage_normalization,
            file_stats,
        )

        if self.n_timebins is not None and time < self.n_timebins:
            pad_amount = self.n_timebins - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad_amount)), mode='constant')

        arr = normalize_spectrogram_numpy(
            arr,
            self.normalization,
            mean=self.mean,
            std=self.std,
        )

        spec = torch.from_numpy(arr).unsqueeze(0).to(dtype=self.output_torch_dtype)

        return spec, filename 

    def __len__(self):
        return len(self.file_dirs)

class SupervisedSpectogramDataset(Dataset):
    def __init__(
        self,
        dir,
        annotation_file_path,
        n_timebins=1024,
        mode="detect",
        white_noise=0.0,
        audio_params_override=None,
        normalization="audio_params",
        output_dtype="float32",
    ):
        """
        n_timebins = None means no cropping
        white_noise: standard deviation of white noise to add after normalization (0.0 = no noise)
        """
        self.file_dirs = sorted(list(Path(dir).glob("*.npy")))

        # Load audio parameters using utility function (or override with pretrain params)
        self.audio_data_json = load_audio_params(dir)
        if audio_params_override is not None:
            merged_audio_params = dict(self.audio_data_json)
            merged_audio_params.update(audio_params_override)
            self.audio_data_json = merged_audio_params
        
        self.n_mels = self.audio_data_json["mels"]
        self.sr = self.audio_data_json["sr"]
        self.hop_size = self.audio_data_json["hop_size"]
        self.fft = self.audio_data_json["fft"]
        self.mean = self.audio_data_json["mean"]
        self.std = self.audio_data_json["std"]
        self.n_timebins = n_timebins
        self.mean = np.float32(self.mean)
        self.std = np.float32(self.std)
        self.normalization = normalization
        self.output_dtype = output_dtype

        self.mode = mode ## detect = vocalization present/absent, unit_detect = syllable present/absent, classify = syllable class
        self.annotation_file_path = annotation_file_path
        self.white_noise = white_noise
        assert self.normalization in SUPPORTED_NORMALIZATIONS
        self.output_torch_dtype = resolve_output_dtype(output_dtype)
        self.storage_dtype = self.audio_data_json.get("storage_dtype", "float32")
        self.storage_normalization = self.audio_data_json.get("storage_normalization", "none")
        assert self.storage_dtype in SUPPORTED_STORAGE_DTYPES
        assert self.storage_normalization in SUPPORTED_STORAGE_NORMALIZATIONS
        self.quantization_manifest = load_quantization_manifest(dir, self.audio_data_json)

        with open(annotation_file_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        self.class_id_map = get_class_id_map_from_annotations(annotation_file_path, mode)
        self._label_index = self._build_label_index(
            annotations,
            mode,
            class_id_map=self.class_id_map,
        )
        
        # Automatically determine number of classes from annotations
        self.num_classes = get_num_classes_from_annotations(annotation_file_path, mode)

        if len(self.file_dirs) == 0:
            raise SystemExit("no files!!")

    def ms_to_timebins(self, ms_value):
        """
        purpose: converts ms value to timebin value 

        the formula of converting ms to timebins:
        time_bin = (time_ms / 1000) × sample_rate / hop_length

        audio_params (tuple) = sr, n_mels, hop_size, fft
        """

        # int rounding is floor rounding ... could be a point of error 
        return int((ms_value / 1000) * self.sr / self.hop_size)

    @staticmethod
    def _build_label_index(annotations, mode, class_id_map=None):
        if mode not in ["detect", "classify", "unit_detect"]:
            raise ValueError("mode must be 'detect', 'classify', or 'unit_detect'")

        label_index = {}
        for rec in annotations.get("recordings", []):
            rec_filename = Path(rec["recording"]["filename"]).stem
            events = rec.get("detected_events", [])
            if mode == "detect":
                labels = [
                    {"onset_ms": event["onset_ms"], "offset_ms": event["offset_ms"]}
                    for event in events
                ]
            elif mode == "classify":
                labels = []
                for event in events:
                    for unit in event.get("units", []):
                        unit_id = unit.get("id")
                        if unit_id is None:
                            continue
                        unit_id = int(unit_id)
                        mapped_id = class_id_map.get(unit_id) if class_id_map is not None else unit_id
                        if mapped_id is None:
                            continue
                        remapped = dict(unit)
                        remapped["id"] = int(mapped_id)
                        labels.append(remapped)
            else:
                labels = [unit for event in events for unit in event.get("units", [])]
            label_index[rec_filename] = labels
        return label_index

    def create_label_array(self, labels, start_bin, end_bin):
        """
        Create a 1D array of labels matching the spectrogram time dimension
        """
        window_len = end_bin - start_bin
        if window_len <= 0:
            return np.zeros(0, dtype=np.int64)
        
        # Initialize label array with silence (class 0)
        if self.mode in ["detect", "unit_detect"]:
            label_arr = np.full(window_len, 0, dtype=np.int64)  # 0 = silence
        else:  # classify
            label_arr = np.full(window_len, 0, dtype=np.int64)  # 0 = silence
        
        # Fill in labels based on onset/offset
        for label in labels:
            onset_bin = self.ms_to_timebins(label["onset_ms"])
            offset_bin = self.ms_to_timebins(label["offset_ms"])
            if offset_bin <= start_bin or onset_bin >= end_bin:
                continue
            onset_bin = max(onset_bin, start_bin) - start_bin
            offset_bin = min(offset_bin, end_bin) - start_bin
            
            if self.mode in ["detect", "unit_detect"]:
                label_arr[onset_bin:offset_bin] = 1  # 1 = present (vocalization or unit)
            else:  # classify
                label_arr[onset_bin:offset_bin] = label["id"] + 1  # shift by +1, so classes are 1, 2, 3, ...
        
        return label_arr
            
    def __getitem__(self, index):
        path = self.file_dirs[index]  

        filename = path.stem
        arr = np.load(path, mmap_mode="r")
        time = arr.shape[1]
        file_stats = self.quantization_manifest.get(path.name)

        base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(filename)
        labels = self._label_index.get(base_filename)
        if labels is None:
            raise ValueError(f"No matching recording found for: {base_filename}")
        labels = clip_labels_to_chunk(labels, chunk_start_ms, chunk_end_ms)

        # we want to load the whole file if n_timebins is None
        if self.n_timebins is None:
            start = 0
            end = time
            arr = arr[:, start:end]

        # loading a chunk 
        else:
            # crop 
            if time > self.n_timebins:
                start = random.randint(0, time - self.n_timebins)
                end = start + self.n_timebins
                arr = arr[:, start:end]
            elif time == self.n_timebins:
                start = 0
                end = time
                arr = arr[:, start:end]
            else:
                start = 0
                end = time
                arr = arr[:, :]

        # Create label array matching spectrogram time dimension
        labels = self.create_label_array(labels, start, end)

        arr = decode_storage_to_raw(
            arr,
            self.storage_dtype,
            self.storage_normalization,
            file_stats,
        )

        # Crop/pad spectrograms and labels
        if self.n_timebins is not None:
            if time < self.n_timebins:
                pad_amount = self.n_timebins - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_amount)), mode='constant')
                labels = np.pad(labels, (0, pad_amount), mode='constant', constant_values=0)  # pad with silence (class 0)

        arr = normalize_spectrogram_numpy(
            arr,
            self.normalization,
            mean=self.mean,
            std=self.std,
        )

        # Apply white noise augmentation if enabled
        if self.white_noise > 0.0:
            noise = np.random.normal(0, self.white_noise, arr.shape).astype(np.float32)
            arr += noise

        spec = torch.from_numpy(arr).unsqueeze(0).to(dtype=self.output_torch_dtype)
        labels = torch.from_numpy(labels)

        return spec, labels, filename

    def __len__(self):
        return len(self.file_dirs)
