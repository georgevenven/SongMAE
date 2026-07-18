import argparse
import subprocess
import sys
import unittest

import numpy as np

from src.embeddings.syllable_knn import add_args, standardize


class SyllableKnnTests(unittest.TestCase):
    def test_zscore_uses_reference_statistics(self):
        reference = np.array([[1, 10], [3, 14]], dtype=np.float32)
        query = np.array([[5, 18]], dtype=np.float32)
        reference, query = standardize(reference, query)
        np.testing.assert_allclose(reference.mean(axis=0), 0)
        np.testing.assert_allclose(reference.std(axis=0), 1)
        np.testing.assert_allclose(query, [[3, 3]])

    def test_end_of_block_protocol(self):
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
        self.assertEqual(args.pca_components, 128)
        for name in ("pca_whiten", "zscore", "reuse", "max_points"):
            self.assertFalse(hasattr(args, name))

    def test_import_does_not_load_umap_or_sklearn(self):
        code = (
            "import sys; import src.embeddings.syllable_knn; "
            "assert 'umap' not in sys.modules; assert 'sklearn' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
