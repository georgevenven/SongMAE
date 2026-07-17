import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.evals.AVEX.birdset import manifest_paths, normalize_empty_labels
from src.evals.AVEX.prepare_birdset import label_lookup, local_config
from src.evals.AVEX.windows import pool_embeddings, select_layers, sliding_window_starts


class AvexAdapterTests(unittest.TestCase):
    def test_birdset_manifest_paths_share_train_and_validation(self):
        paths = manifest_paths("/data/birdset")
        self.assertEqual(paths["HSN-train"], "/data/birdset/manifests/HSN-train.jsonl")
        self.assertEqual(paths["HSN-validation"], paths["HSN-train"])
        self.assertEqual(paths["HSN-test"], "/data/birdset/manifests/HSN-test.jsonl")

    def test_birdset_normalizes_legacy_empty_labels(self):
        with TemporaryDirectory() as directory:
            manifests = Path(directory) / "manifests"
            manifests.mkdir()
            test = manifests / "HSN-test.jsonl"
            test.write_text(
                '{"path": "empty.wav", "labels_as_list": ["None"]}\n'
                '{"path": "mixed.wav", "labels_as_list": ["bird", "None"]}\n'
                '{"path": "bird.wav", "labels_as_list": ["bird"], "note": "None"}\n'
            )
            original = test.read_text()
            self.assertEqual(normalize_empty_labels(directory), 0)
            self.assertEqual(test.read_text(), original)
            (manifests / "HSN-test.complete").write_text('{"rows": 3}\n')

            self.assertEqual(normalize_empty_labels(directory), 2)
            self.assertEqual(normalize_empty_labels(directory), 0)
            rows = [json.loads(line) for line in test.read_text().splitlines()]
            self.assertEqual(
                [row["labels_as_list"] for row in rows],
                [[], ["bird"], ["bird"]],
            )
            self.assertIn('"note": "None"', test.read_text())

    def test_birdset_drops_legacy_empty_training_rows(self):
        with TemporaryDirectory() as directory:
            manifests = Path(directory) / "manifests"
            manifests.mkdir()
            train = manifests / "HSN-train.jsonl"
            train.write_text(
                '{"path": "empty.wav", "labels_as_list": ["None"], "stratify_label": "None"}\n'
                '{"path": "bird.wav", "labels_as_list": ["None", "bird"], "stratify_label": "None"}\n'
            )
            complete = manifests / "HSN-train.complete"
            complete.write_text('{"rows": 2}\n')

            self.assertEqual(normalize_empty_labels(directory), 2)
            self.assertEqual(
                train.read_text(),
                '{"path": "bird.wav", "labels_as_list": ["bird"], "stratify_label": "bird"}\n',
            )
            self.assertEqual(json.loads(complete.read_text())["rows"], 1)

    def test_birdset_label_lookup_filters_only_legacy_missing_class(self):
        labels = {"Passeriformes Corvidae Cyanocorax morio": 0, "None": 1}
        taxonomy = {
            "brnjay": {"sci_name": "Psilorhinus morio"},
            "runwre1": {"sci_name": "Campylorhynchus rufinucha"},
        }
        self.assertEqual(label_lookup(labels, taxonomy, list(taxonomy)), {"brnjay": next(iter(labels)), "runwre1": None})

    def test_birdset_config_uses_scalar_stratification(self):
        split = {
            "sample_rate": 16000,
            "data_root": "/private",
            "transformations": [{"type": "train_val_split", "stratify_column": "labels_as_list"}],
        }
        config = local_config({"train": split, "validation": split, "test": split}, "/data/birdset")
        for value in config["evaluation_sets"][0].values():
            if isinstance(value, dict):
                self.assertEqual(value["sample_rate"], 32000)
                self.assertEqual(value["data_root"], "/data/birdset")
                self.assertEqual(value["transformations"][0]["stratify_column"], "stratify_label")

    def test_sliding_windows_use_half_overlap_and_include_tail(self):
        self.assertEqual(sliding_window_starts(3, 8), [0])
        self.assertEqual(sliding_window_starts(16, 8), [0, 4, 8])
        self.assertEqual(sliding_window_starts(17, 8), [0, 4, 8, 9])

    def test_layer_selectors(self):
        self.assertEqual(select_layers(["last_layer"], 4), [3])
        self.assertEqual(select_layers(["all"], 4), [0, 1, 2, 3])
        self.assertEqual(select_layers([0, "layer_2", -1, "layer_2"], 4), [0, 2, 3])

    def test_pooling_uses_every_valid_token_across_windows(self):
        embeddings = torch.tensor(
            [
                [
                    [[2, 4], [100, 100], [100, 100]],
                    [[102, 104], [200, 200], [200, 200]],
                ],
                [
                    [[6, 8], [10, 12], [14, 16]],
                    [[106, 108], [110, 112], [114, 116]],
                ],
            ],
            dtype=torch.float32,
        )
        pooled = pool_embeddings(embeddings, torch.tensor([1, 3]))
        torch.testing.assert_close(pooled, torch.tensor([[8, 10], [108, 110]], dtype=torch.float32))

    def test_pooling_ignores_empty_windows(self):
        embeddings = torch.tensor(
            [
                [[[2.0], [4.0]]],
                [[[100.0], [200.0]]],
            ]
        )
        pooled = pool_embeddings(embeddings, torch.tensor([2, 0]))
        torch.testing.assert_close(pooled, torch.tensor([[3.0]]))


if __name__ == "__main__":
    unittest.main()
