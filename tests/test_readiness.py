import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from project_code.algorithms.loss_functions import MaskedMSELoss
from project_code.data.load_data import load_col_types
from project_code.train.run_baseline import ROOT, check_scalers, make_model, prepare, validate_frames


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ROOT / "data/processed/init_net"
        cls.types = load_col_types("init_net", cls.dataset.parent)
        cls.full = pd.read_csv(cls.dataset / "init_net.csv", index_col=0)
        cls.frames = {s: pd.read_csv(cls.dataset / f"{s}.csv", index_col=0)
                      for s in ("train", "val", "test")}
        cls.training = json.loads((ROOT / "configs/hc_initnet.json").read_text())["training"]

    def reject(self, mutate, message):
        frames = {s: d.copy() for s, d in self.frames.items()}
        mutate(frames)
        with self.assertRaisesRegex(ValueError, message):
            validate_frames(frames, self.full, self.types)

    def test_original_partition_integrity(self):
        validate_frames(self.frames, self.full, self.types)
        self.assertEqual({k: len(v) for k, v in self.frames.items()},
                         {"train": 1647, "val": 353, "test": 353})

    def test_missing_column_rejected(self):
        self.reject(lambda f: f["train"].drop(columns=[self.types["input"]["all"][0]], inplace=True),
                    "missing columns")

    def test_duplicate_id_rejected(self):
        def mutate(f):
            f["train"] = pd.concat([f["train"], f["train"].iloc[:1]])
        self.reject(mutate, "duplicate")

    def test_overlap_rejected(self):
        def mutate(f):
            f["val"] = pd.concat([f["val"], f["train"].iloc[:1]])
        self.reject(mutate, "overlap")

    def test_nonfinite_eligible_output_rejected(self):
        col = self.types["output"]["all"][0]
        def mutate(f):
            idx = f["train"].index[f["train"][f"estim_{col}"]][0]
            f["train"].loc[idx, col] = np.nan
        self.reject(mutate, "non-finite")

    def test_nonpositive_log_output_rejected(self):
        col = self.types["output"]["log"][0]
        self.reject(lambda f: f["train"].__setitem__(col, 0), "non-positive")

    def test_full_table_disagreement_rejected(self):
        col = self.types["input"]["all"][0]
        self.reject(lambda f: f["train"].__setitem__(col, f["train"][col] + 1), "differs")

    def test_invalid_mask_rejected(self):
        col = self.types["mask"]["all"][0]
        self.reject(lambda f: f["train"].__setitem__(col, 2), "eligibility mask")

    def test_upstream_masked_loss_hand_calculation(self):
        pred = torch.tensor([[1., 100.], [3., 5.]])
        target = torch.tensor([[0., 0.], [1., 1.]])
        mask = torch.tensor([[1., 0.], [1., 1.]])
        self.assertEqual(MaskedMSELoss()(pred, target, mask).item(), 7.)

    def test_train_only_scalers_and_roundtrip(self):
        _, tensors, _, scalers, _ = prepare(self.dataset, 1)
        check_scalers(tensors, scalers)
        # Held-out perturbations must not change fitted training statistics.
        from project_code.data.prepare_data_pytorch import log_standardize_data
        changed = {s: d.copy() for s, d in self.frames.items()}
        changed["val"][self.types["input"]["log"]] *= 100
        _, alternate, _ = log_standardize_data(changed, self.types)
        for kind in ("input", "output"):
            torch.testing.assert_close(alternate[kind].mean, scalers[kind].mean)
            torch.testing.assert_close(alternate[kind].std, scalers[kind].std)

    def test_architecture_matches_trusted_upstream_checkpoint(self):
        types, tensors, _, scalers, _ = prepare(self.dataset, 1)
        model = make_model(self.training, types, scalers)
        # Whole-object pickle is loaded only for this trusted, pinned source artifact.
        reference = torch.load(ROOT / "results/init_net/models/logQ_81327e-01_MLP.pth",
                               map_location="cpu", weights_only=False).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        x = scalers["input"].transform(tensors["val"]["input"])
        with torch.no_grad():
            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
