import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.embeddings.syllable_knn import add_args, occurrence_neighbors, occurrences, prepare


class SyllableKnnTests(unittest.TestCase):
    def test_zscore_uses_reference_tokens(self):
        reference, query = prepare(
            np.array([[1, 10], [3, 14]], dtype=np.float32),
            np.array([[5, 18]], dtype=np.float32),
            0,
            42,
        )
        np.testing.assert_allclose(reference, [[-2**-0.5] * 2, [2**-0.5] * 2])
        np.testing.assert_allclose(query, [[2**-0.5] * 2])

    def test_neighbors_exclude_query_event_and_duplicate_occurrences(self):
        rows = [
            {"event": 0},
            {"event": 1},
            {"event": 0},
            {"event": 2},
        ]
        candidates = np.array([[0, 1, 2, 3, 4]])
        reference_occurrences = np.array([0, 1, 1, 2, 3])
        neighbors = occurrence_neighbors(candidates, np.array([0]), reference_occurrences, rows, 2)
        np.testing.assert_array_equal(neighbors, [[1, 3]])

    def test_overlapping_token_belongs_to_syllable_and_silence(self):
        store = {
            "token_start_ms": np.array([0]),
            "token_end_ms": np.array([20]),
            "recording_stem": np.array(["recording"]),
            "song_id": np.array([0]),
        }
        annotations = {
            "recordings": [{
                "recording": {"filename": "recording.wav", "bird_id": "bird"},
                "detected_events": [{
                    "onset_ms": 0,
                    "offset_ms": 20,
                    "units": [{"onset_ms": 5, "offset_ms": 10, "id": 2}],
                }],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(annotations))
            rows = occurrences(store, path, "bird")
        self.assertEqual([row["label"] for row in rows], [0, 3, 0])
        self.assertTrue(all(row["tokens"].tolist() == [0] for row in rows))

    def test_protocol_defaults(self):
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([
            "--spec_dir", "specs",
            "--annotation_file", "annotations.json",
            "--out_dir", "out",
            "--model", "songmae",
            "--bird", "bird",
        ])
        self.assertEqual(args.target_feature_type, "end_of_block")
        self.assertEqual(args.encoder_layer_idx, -1)
        self.assertEqual(args.pca_components, 0)
        self.assertEqual(args.k_values, "1,5,10")

    def test_import_does_not_load_umap_or_sklearn(self):
        code = (
            "import sys; import src.embeddings.syllable_knn; "
            "assert 'umap' not in sys.modules; assert 'sklearn' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
