import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.core.model import SongMAE
from src.evals.syllable_finetune import (
    Example,
    FineTuneSplit,
    SongMAEFinetuner,
    build_finetune_split,
    fit,
    patch_labels,
    token_spans,
)


def split_args(seconds="MAX"):
    return SimpleNamespace(test_fraction=0.2, dev_fraction=0.2, seed=42, max_train_seconds=seconds)


def tiny_run(tmp_path):
    config = {
        "patch_size": [4, 2],
        "patch_height": 4,
        "patch_width": 2,
        "mels": 8,
        "num_timebins": 8,
        "mask_p": 0.5,
        "mask_c": 0.1,
        "mask_type": "random",
        "qk_norm": True,
        "dropout": 0.0,
        "enc_hidden_d": 8,
        "enc_n_head": 2,
        "enc_n_layer": 2,
        "enc_dim_ff": 16,
        "dec_hidden_d": 4,
        "dec_n_head": 1,
        "dec_n_layer": 1,
        "dec_dim_ff": 8,
    }
    weights = tmp_path / "weights"
    weights.mkdir()
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(SongMAE(config).state_dict(), weights / "model_step_000000.pth")
    return tmp_path


class SyllableFinetuneTest(unittest.TestCase):
    def test_tune_dev_test_groups_are_disjoint(self):
        examples, units = [], {}
        for index in range(6):
            stem = f"song{index}"
            examples.append(Example(torch.zeros(1), np.array([-1, 0]), f"{stem}:{index}", stem, 0, 1000))
            units[stem] = [(100, 300, 1), (500, 700, 2)]

        split, classes = build_finetune_split(examples, units, split_args())
        train, tune, dev, test = map(
            set,
            (split.train_groups, split.tune_groups, split.dev_groups, split.test_groups),
        )

        self.assertFalse(train & test)
        self.assertFalse(tune & dev)
        self.assertFalse(tune & test)
        self.assertFalse(dev & test)
        self.assertEqual(tune | dev | test, {row.group for row in examples})
        self.assertEqual(classes, [0, 1, 2])
        self.assertEqual(split.missing_dev_classes, [])

    def test_rare_test_class_is_not_moved_to_dev(self):
        examples, units = [], {}
        for index in range(6):
            stem = f"song{index}"
            examples.append(Example(torch.zeros(1), np.array([-1, 0]), f"{stem}:{index}", stem, 0, 1000))
            units[stem] = [(100, 300, 1)] + ([(500, 700, 2)] if index < 2 else [])

        split, _ = build_finetune_split(examples, units, split_args())

        self.assertIn("song0:0", split.tune_groups)
        self.assertNotIn("song0:0", split.dev_groups)
        self.assertEqual(split.missing_dev_classes, [2])

    def test_patch_labels_and_partial_span_alignment(self):
        labels = np.array([-1, 0, -1, -1, 2, -1])
        example = Example(torch.zeros(1), labels, "song:0", "song", 100, 130)

        self.assertEqual(patch_labels(labels, 4).tolist(), [1, 3])
        self.assertEqual(token_spans(example, 2, patch_width=4), [("song", 100, 120), ("song", 120, 130)])

    def test_all_blocks_finetuning_freezes_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(songmae_run_dir=tiny_run(Path(directory)), checkpoint=None)
            model = SongMAEFinetuner(args, [0, 1, 2])
            example = Example(torch.randn(1, 8, 8), np.array([-1, 0, 0, -1, 1]), "song:0", "song", 0, 25)

            model.train_mode()
            batch = model.forward_examples([example])
            batch.logits.sum().backward()

            self.assertEqual(batch.logits.shape, (3, 3))
            self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.encoder.layers[0].parameters()))
            self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.encoder.layers[-1].parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in model.backbone.patch_projection.parameters()))
            self.assertTrue(any(parameter.grad is not None for parameter in model.head.parameters()))

    def test_fit_selects_on_dev_and_tests_once(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 1)
                self.head = torch.nn.Linear(1, 1)
                self.marker = None

        examples = [
            Example(torch.zeros(1), np.array([-1]), group, group, 0, 10)
            for group in ("train", "tune", "dev", "test")
        ]
        split = FineTuneSplit(
            train_groups=["train"],
            tune_groups=["tune"],
            dev_groups=["dev"],
            test_groups=["test"],
            train_order=["train"],
            train_seconds=1,
            tune_seconds=1,
            dev_seconds=1,
            test_seconds=1,
            missing_dev_classes=[],
        )
        args = SimpleNamespace(
            seed=42,
            encoder_lrs=[1e-5, 1e-4],
            head_lr=1e-3,
            weight_decay=0.01,
            epochs=2,
            fixed=False,
        )
        test_calls = []

        def fake_train(model, rows, classes, args, optimizer, scaler, epoch):
            model.marker = (optimizer.param_groups[0]["lr"], epoch)
            return float(epoch)

        def fake_evaluate(model, rows, classes, units, args):
            if rows[0].group == "test":
                test_calls.append(model.marker)
                return {"macro_fer": 0.5}
            scores = {(1e-5, 1): 0.4, (1e-5, 2): 0.35, (1e-4, 1): 0.3, (1e-4, 2): 0.32}
            return {"macro_fer": scores[model.marker]}

        with (
            patch("src.evals.syllable_finetune.make_model", side_effect=lambda *args: FakeModel()),
            patch("src.evals.syllable_finetune.train_epoch", side_effect=fake_train),
            patch("src.evals.syllable_finetune.evaluate", side_effect=fake_evaluate),
        ):
            metrics = fit(args, examples, split, [0], {}, torch.device("cpu"))

        self.assertEqual(metrics["selected_encoder_lr"], 1e-4)
        self.assertEqual(metrics["selected_epoch"], 1)
        self.assertEqual(test_calls, [(1e-4, 1)])

    def test_fit_fixed_uses_all_epochs_without_dev(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 1)
                self.head = torch.nn.Linear(1, 1)

        examples = [Example(torch.zeros(1), np.array([-1]), group, group, 0, 10) for group in ("train", "test")]
        split = FineTuneSplit(["train"], [], [], ["test"], ["train"], 1, 0, 0, 1, [])
        args = SimpleNamespace(
            seed=42,
            encoder_lrs=[5e-5],
            head_lr=1e-3,
            weight_decay=0.01,
            epochs=3,
            fixed=True,
        )
        epochs = []

        with (
            patch("src.evals.syllable_finetune.make_model", side_effect=lambda *args: FakeModel()),
            patch("src.evals.syllable_finetune.train_epoch", side_effect=lambda *args: epochs.append(args[-1]) or 0.0),
            patch("src.evals.syllable_finetune.evaluate", return_value={"macro_fer": 0.5}),
        ):
            metrics = fit(args, examples, split, [0], {}, torch.device("cpu"))

        self.assertEqual(epochs, [1, 2, 3])
        self.assertEqual(metrics["selected_encoder_lr"], 5e-5)
        self.assertEqual(metrics["selected_epoch"], 3)
        self.assertEqual(metrics["selection_mode"], "fixed")
        self.assertIsNone(metrics["dev_macro_fer"])


if __name__ == "__main__":
    unittest.main()
