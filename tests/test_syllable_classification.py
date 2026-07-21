import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core.embedding_store import save_embedding_arrays
from src.evals.syllable_classification import load_embeddings, make_folds, standardize


class SyllableClassificationTests(unittest.TestCase):
    def test_loads_final_layer(self):
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
            x, y, spans, groups = load_embeddings(path)
            np.testing.assert_array_equal(x, layers[:, -1])
            self.assertEqual(y.tolist(), [1, 2, 0])
            self.assertEqual(spans, [("a", 0, 10), ("a", 10, 20), ("b", 0, 10)])
            self.assertEqual(groups, ["a:0", "a:0", "b:1"])

    def test_folds_are_song_disjoint(self):
        groups = np.repeat([f"song_{index}" for index in range(6)], 2).tolist()
        folds = make_folds(np.tile([0, 1], 6), groups, count=3, seed=42)
        validation = []
        for fold in folds:
            train = set(fold["train_groups"])
            val = set(fold["val_groups"])
            self.assertTrue(train.isdisjoint(val))
            validation.extend(val)
        self.assertEqual(len(validation), len(set(validation)))

    def test_standardizes_from_training_only(self):
        x = np.array([[0, 0], [2, 4], [100, 200]], dtype=np.float32)
        train, val = standardize(x, np.array([0, 1]), np.array([2]))
        np.testing.assert_allclose(train.mean(axis=0), 0)
        np.testing.assert_allclose(train.std(axis=0), 1)
        np.testing.assert_allclose(val, [[99, 99]])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            embeddings = directory / "embeddings"
            stems = np.repeat([f"song_{index}" for index in range(6)], 2)
            features = np.tile([[1, 0, 0, 0], [0, 1, 0, 0]], (6, 1)).astype(np.float32)
            save_embedding_arrays(
                embeddings,
                {
                    "encoded_embeddings": features,
                    "labels_downsampled": np.tile([0, -1], 6),
                    "recording_stem": stems,
                    "token_start_ms": np.tile([0, 10], 6),
                    "token_end_ms": np.tile([10, 20], 6),
                    "song_id": np.zeros(12, dtype=np.int64),
                },
            )
            annotations = directory / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "recordings": [
                            {
                                "recording": {"filename": f"song_{index}.wav"},
                                "detected_events": [
                                    {
                                        "units": [
                                            {"onset_ms": 0, "offset_ms": 10, "id": 0}
                                        ]
                                    }
                                ],
                            }
                            for index in range(6)
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
                    "--pca_components",
                    "2",
                    "--max_iter",
                    "100",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(result.stdout)
            self.assertEqual(result["folds"], 3)
            self.assertEqual(result["pca_fit_scope"], "all_extracted_tokens")
            self.assertEqual(result["standardization_fit_scope"], "training_fold_after_pca")
            self.assertEqual(result["frames"], 120)

            capped = subprocess.run(
                [
                    sys.executable,
                    str(root / "src/evals/syllable_classification_capped.py"),
                    "--embeddings",
                    str(embeddings),
                    "--annotations",
                    str(annotations),
                    "--label_cap",
                    "1",
                    "--pca_components",
                    "2",
                    "--max_iter",
                    "100",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            capped = json.loads(capped.stdout)
            self.assertEqual(capped["label_cap"], 1)
            self.assertEqual(capped["label_budget"], "at_most_occurrences_per_class")
            self.assertEqual(capped["frames"], 120)


if __name__ == "__main__":
    unittest.main()
