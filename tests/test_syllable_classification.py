import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.core.embedding_store import save_embedding_arrays
from src.evals.syllable_classification import (
    ScalarMixMLP,
    feature_stats,
    fit_torch_classifier,
    load_embeddings,
    predict_torch_classifier,
)


class ScalarMixTests(unittest.TestCase):
    def test_uniform_initialization(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        expected = x.mean(dim=1)
        model = ScalarMixMLP(3, 4, 2)
        model.head = torch.nn.Identity()
        torch.testing.assert_close(model(x), expected)

    def test_layer_embedding_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings"
            layers = np.arange(24, dtype=np.float16).reshape(3, 2, 4)
            save_embedding_arrays(
                path,
                {
                    "encoded_embeddings": layers,
                    "labels_downsampled": np.array([0, 1, -1]),
                    "recording_stem": np.array(["a", "a", "b"]),
                    "token_start_ms": np.array([0, 10, 0]),
                    "token_end_ms": np.array([10, 20, 10]),
                    "song_id": np.array([0, 0, 1]),
                },
            )
            x, y, spans, groups = load_embeddings(path, all_layers=True)
            self.assertEqual(x.shape, (3, 2, 4))
            self.assertEqual(y.tolist(), [1, 2, 0])
            self.assertEqual(spans, [("a", 0, 10), ("a", 10, 20), ("b", 0, 10)])
            self.assertEqual(groups, ["a:0", "a:0", "b:1"])
            final, _, _, _ = load_embeddings(path)
            np.testing.assert_array_equal(final, layers[:, -1])

    def test_fit_and_predict(self):
        x = np.random.default_rng(0).normal(size=(8, 3, 4)).astype(np.float16)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        indices = np.arange(y.size)
        args = SimpleNamespace(model="scalar_mix", seed=0, lr=1e-3, epochs=2, batch_size=4)
        model, classes, mean, std = fit_torch_classifier(x, y, indices, args)
        predictions = predict_torch_classifier(model, classes, x, indices, args.batch_size, mean, std)
        self.assertEqual(predictions.shape, y.shape)
        self.assertEqual(classes.tolist(), [0, 1])
        self.assertAlmostEqual(float(model.weights.softmax(dim=0).sum()), 1.0, places=6)

    def test_mlp_uses_shared_torch_loop(self):
        x = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        indices = np.arange(y.size)
        args = SimpleNamespace(model="mlp", seed=0, lr=1e-3, epochs=1, batch_size=4)
        model, classes, mean, std = fit_torch_classifier(x, y, indices, args)
        predictions = predict_torch_classifier(model, classes, x, indices, args.batch_size, mean, std)
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(next(model.parameters()).device.type, expected_device)
        self.assertEqual(predictions.shape, y.shape)

    def test_final_layer_standardization_matches_mlp(self):
        x = np.random.default_rng(0).normal(size=(8, 3, 4)).astype(np.float32)
        indices = np.arange(x.shape[0])
        layer_mean, layer_std = feature_stats(x, indices, batch_size=3)
        final_mean, final_std = feature_stats(x[:, -1], indices, batch_size=3)
        np.testing.assert_allclose(layer_mean[-1], final_mean)
        np.testing.assert_allclose(layer_std[-1], final_std)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            embeddings = directory / "embeddings"
            stems = np.repeat(["a", "b", "c", "d"], 2)
            save_embedding_arrays(
                embeddings,
                {
                    "encoded_embeddings": np.random.default_rng(0).normal(size=(8, 3, 4)).astype(np.float16),
                    "labels_downsampled": np.tile([0, -1], 4),
                    "recording_stem": stems,
                    "token_start_ms": np.tile([0, 10], 4),
                    "token_end_ms": np.tile([10, 20], 4),
                    "song_id": np.zeros(8, dtype=np.int64),
                },
            )
            annotations = directory / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "recordings": [
                            {
                                "recording": {"filename": f"{stem}.wav"},
                                "detected_events": [
                                    {
                                        "onset_ms": 0,
                                        "offset_ms": 20,
                                        "units": [{"onset_ms": 0, "offset_ms": 10, "id": 0}],
                                    }
                                ],
                            }
                            for stem in "abcd"
                        ]
                    }
                )
            )
            root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "src/evals/syllable_classification.py"),
                    "--embeddings",
                    str(embeddings),
                    "--annotations",
                    str(annotations),
                    "--model",
                    "scalar_mix",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "4",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            metrics = json.loads(result.stdout)
            self.assertEqual(metrics["encoder_scope"], "frozen_scalar_mix")
            self.assertEqual(metrics["classifier"], "mlp_1024_256")
            self.assertEqual(len(metrics["layer_weights"]), 3)


if __name__ == "__main__":
    unittest.main()
