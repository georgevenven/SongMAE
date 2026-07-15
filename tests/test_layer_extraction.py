import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.core.extract_embedding import _extract_segment_arrays
from src.external_models.aves import extract_features as extract_aves
from src.external_models.data_loader import (
    convolution_feature_map,
    convolution_geometry,
    save_concatenated_embeddings,
    WavFromSpectrogramDataset,
)
from src.external_models.hubert import extract_features as extract_hubert


class FakeSongMAE:
    def forward_encoder_inference(self, x, encoder_layer_idx=None, return_all_layers=False, **_):
        base = torch.arange(x.shape[0] * 6 * 2, device=x.device).reshape(x.shape[0], 6, 2)
        layers = torch.stack([base + 100 * index for index in range(3)], dim=1)
        index = -1 if encoder_layer_idx is None else encoder_layer_idx
        return (layers if return_all_layers else layers[:, index]), base


class FakeAves:
    def extract_features(self, wav, lengths):
        base = torch.arange(24, device=wav.device).reshape(1, 3, 8)
        return [base, base + 100], torch.tensor([3], device=wav.device)


class FakeFeatureExtractor:
    def __call__(self, values, **_):
        return {"input_values": torch.as_tensor(values).unsqueeze(0)}


class FakeHubert:
    def __call__(self, **_):
        base = torch.arange(24).reshape(1, 3, 8)
        return SimpleNamespace(last_hidden_state=base + 200, hidden_states=(base, base + 100, base + 200))


class FakeSpecDataset:
    samples = [(Path("recording.npy"), {"on_timebins": -1, "off_timebins": 4})]

    def __getitem__(self, index):
        return None, torch.arange(4), "recording"


class LayerExtractionTests(unittest.TestCase):
    def test_waveform_crop_uses_clamped_spectrogram_bounds(self):
        dataset = object.__new__(WavFromSpectrogramDataset)
        dataset.spec_dataset = FakeSpecDataset()
        dataset.wav_paths = {"recording": Path("recording.wav")}
        dataset.audio_params = (32000, 128, 160, 1024)
        item = dataset[0]
        self.assertEqual((item["start_ms"], item["end_ms"]), (0, 20))

    def test_external_store_preserves_original_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = torch.tensor([1, 2, 3, 4])
            item = {
                "labels": labels,
                "start_ms": 0,
                "end_ms": 20,
                "recording_stem": "recording",
                "song_id": 0,
                "spec_path": "recording.npy",
                "wav_path": "recording.wav",
            }
            row = {
                "item": item,
                "encoded_embeddings": np.zeros((2, 3), dtype=np.float32),
                "labels_downsampled": np.array([1, 3]),
                "token_edges": np.array([0, 2, 4]),
            }
            save_concatenated_embeddings(directory, [row])
            np.testing.assert_array_equal(np.load(Path(directory) / "labels_original.npy"), labels.numpy())

    def test_wav2vec_token_alignment(self):
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        geometry = convolution_geometry(kernels, strides, samples_per_timebin=80)
        self.assertEqual(geometry, (2.5, 4.0))

        labels = np.repeat(np.arange(250), 4)
        mapped, edges = convolution_feature_map(labels, output_length=249, geometry=geometry)
        np.testing.assert_array_equal(mapped, np.arange(249))
        np.testing.assert_array_equal(edges[:3], [0, 4.5, 8.5])
        np.testing.assert_array_equal(edges[-2:], [992.5, 1000])

    def test_songmae_final_layer_matches_layer_stack(self):
        args = dict(
            spec_segment=np.zeros((4, 8), dtype=np.float32),
            labels_segment=np.arange(8, dtype=np.int64),
            model=FakeSongMAE(),
            device=torch.device("cpu"),
            model_num_timebins=12,
            patch_width=4,
            num_patches_height=2,
            num_patches_time=3,
            encoder_layer_idx=None,
            target_feature_type="end_of_block",
        )
        final = _extract_segment_arrays(**args, all_layers=False)
        layers = _extract_segment_arrays(**args, all_layers=True)
        self.assertEqual(layers["encoded_embeddings"].shape, (2, 3, 4))
        np.testing.assert_array_equal(final["encoded_embeddings"], layers["encoded_embeddings"][:, -1])
        np.testing.assert_array_equal(final["labels_downsampled"], layers["labels_downsampled"])

    def test_aves_final_layer_matches_layer_stack(self):
        wav = torch.zeros(100)
        final = extract_aves(FakeAves(), wav, None, False, 1, torch.device("cpu"))
        layers = extract_aves(FakeAves(), wav, None, True, 1, torch.device("cpu"))
        self.assertEqual(layers.shape, (3, 2, 8))
        np.testing.assert_array_equal(final, layers[:, -1])

    def test_hubert_final_layer_matches_layer_stack(self):
        wav = torch.zeros(400)
        final = extract_hubert(FakeFeatureExtractor(), FakeHubert(), wav, 16000, None, False, torch.device("cpu"))
        layers = extract_hubert(FakeFeatureExtractor(), FakeHubert(), wav, 16000, None, True, torch.device("cpu"))
        first = extract_hubert(FakeFeatureExtractor(), FakeHubert(), wav, 16000, 0, False, torch.device("cpu"))
        self.assertEqual(layers.shape, (3, 2, 8))
        np.testing.assert_array_equal(final, layers[:, -1])
        np.testing.assert_array_equal(first, layers[:, 0])


if __name__ == "__main__":
    unittest.main()
