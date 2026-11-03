import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from project_code.train import run_baseline as runner
from project_code.train import train_pytorch_model as trainer


class RunnerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads((runner.ROOT / "configs/hc_initnet.json").read_text())
        cls.prepared = runner.prepare(runner.ROOT / cls.config["dataset"], 1)
        cls.types, cls.tensors, _, cls.scalers, _ = cls.prepared
        cls.model = runner.make_model(cls.config["training"], cls.types, cls.scalers).eval()
        cls.payload = {"state_dict": cls.model.state_dict(), "config": cls.config,
                       "column_types": cls.types,
                       "scalers": {k: {a: getattr(cls.scalers[k], a) for a in ("mean", "std")}
                                   for k in ("input", "output")}}

    def checkpoint(self, payload=None):
        buffer = io.BytesIO()
        torch.save(self.payload if payload is None else payload, buffer)
        buffer.seek(0)
        return buffer

    def test_checkpoint_rebuilds_scalers_and_matches_predictions(self):
        restored = runner.restore_checkpoint(self.checkpoint(), self.config, self.types)
        x = self.scalers["input"].transform(self.tensors["val"]["input"])
        with torch.no_grad():
            torch.testing.assert_close(restored(x), self.model(x), rtol=0, atol=0)
            torch.testing.assert_close(restored.output_transformer.inverse_transform(restored(x)),
                                       self.model.output_transformer.inverse_transform(self.model(x)),
                                       rtol=0, atol=0)

    def test_checkpoint_rejects_semantic_config_mismatches(self):
        for key, value in (("clamp_function", "relu"), ("shared_hidden_layers", [64, 64]),
                           ("use_skip_connection", False), ("scaling_type", "quantile")):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.config)
                cfg["training"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    runner.restore_checkpoint(self.checkpoint(), cfg, self.types)

    def test_checkpoint_rejects_column_reordering_and_transform_changes(self):
        for kind, qualifier in (("input", "all"), ("output", "all"), ("mask", "all"),
                                ("input", "log"), ("output", "scale")):
            with self.subTest(kind=kind, qualifier=qualifier):
                types = copy.deepcopy(self.types)
                types[kind][qualifier] = list(reversed(types[kind][qualifier]))
                with self.assertRaisesRegex(ValueError, "column order or transformation"):
                    runner.restore_checkpoint(self.checkpoint(), self.config, types)

    def test_legacy_incomplete_checkpoint_rejected(self):
        payload = dict(self.payload)
        del payload["column_types"]
        with self.assertRaisesRegex(ValueError, "lacks full column metadata"):
            runner.restore_checkpoint(self.checkpoint(payload), self.config, self.types)

    def test_preparation_routes_both_loaders_to_configured_directory(self):
        dataset = Path("/not-written/alternate/init-copy")
        original = runner.ROOT / self.config["dataset"]
        frames = {s: pd.read_csv(original / f"{s}.csv", index_col=0)
                  for s in ("train", "val", "test")}
        full = pd.read_csv(original / "init_net.csv", index_col=0)
        def read_csv(path, **kwargs):
            self.assertEqual(path.parent, dataset)
            return full if path.name == "init_net.csv" else frames[path.stem]
        with patch.object(runner.pd, "read_csv", side_effect=read_csv), \
             patch.object(runner, "load_col_types", return_value=self.types) as columns, \
             patch.object(runner, "load_dataframes", return_value=(frames, self.types)) as load, \
             patch.object(runner, "sha256", return_value="fixture"), \
             patch.object(Path, "stat", return_value=MagicMock(st_size=0)):
            runner.prepare(dataset, 1)
        columns.assert_called_once_with(dataset.name, datasets_folder=dataset.parent)
        load.assert_called_once_with(dataset.name, datasets_folder=dataset.parent)

    def test_runner_uses_same_dataset_for_audit_and_training_and_observed_stats(self):
        cfg = copy.deepcopy(self.config)
        cfg["dataset"] = "data/processed/../alternate/init-copy"
        dataset = (runner.ROOT / cfg["dataset"]).resolve()
        output = MagicMock(spec=Path)
        output.exists.return_value = False
        output.rglob.return_value = []
        trained = []
        def train(config, factory, name, **kwargs):
            self.assertEqual(dataset, kwargs["datasets_folder"] / name)
            self.assertTrue(kwargs["return_training_stats"])
            model = factory(col_types=self.types, input_transformer=self.scalers["input"],
                            output_transformer=self.scalers["output"],
                            **{k: config[k] for k in runner.MODEL_SETTINGS if k != "clamp_function"})
            x = self.scalers["input"].transform(self.tensors["train"]["input"][:1])
            model(x).sum().backward()
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
            trained.append(model)
            return model, pd.Series({"val_loss": 1.0}), {
                "optimizer_steps": 7, "epochs_completed": 1, "selected_epoch": 1}
        with patch.object(runner, "prepare", return_value=copy.deepcopy(self.prepared)) as prepare, \
             patch.object(trainer, "train_neural_network", side_effect=train), \
             patch.object(runner, "restore_checkpoint", side_effect=lambda *a: trained[0].eval()), \
             patch("builtins.print"):
            report = runner.run(cfg, "smoke", output)
        prepare.assert_called_once_with(dataset, cfg["training"]["batch_size"])
        self.assertEqual(report["dataset_directory"], str(dataset))
        self.assertEqual(report["optimizer_updates"], 7)
        self.assertEqual(report["epochs_completed"], 1)
        self.assertEqual(report["epochs_requested"], 2)
        self.assertFalse(report["test_scored"])

    def train_fixture(self, accumulation=0, early_stop=False, stats=True):
        cfg = copy.deepcopy(self.config["training"])
        cfg.update(batch_size=2, max_epochs=2, n_epochs_gradient_descent=accumulation,
                   early_stopping_metric="val_loss" if early_stop else None, patience=1)
        x = torch.tensor([[1.0], [2.0], [3.0]])
        ds = TensorDataset(x, 2 * x, torch.ones_like(x))
        labels = list(trainer.METRIC_LABEL_TO_NAME)
        metric_frame = pd.DataFrame({"fixture": [1.0] * len(labels)}, index=labels)
        with patch.object(trainer, "load_dataframes", return_value=({}, {})), \
             patch.object(trainer, "create_results_directories_for_dataset"), \
             patch.object(trainer, "prepare_data_tensors", return_value=(
                 {"val": {}}, {"train": DataLoader(ds, batch_size=2), "val": []}, {},
                 {"input": None, "output": None})), \
             patch.object(trainer, "evaluate_pytorch_model", side_effect=[1.0, 2.0]), \
             patch.object(trainer, "evaluate_parameter_predictions_on_data", return_value=metric_frame), \
             patch("builtins.print"):
            return trainer.train_neural_network(cfg, lambda **kw: torch.nn.Linear(1, 1),
                                                "fixture", device=torch.device("cpu"),
                                                return_training_stats=stats)

    def test_counts_real_steps_with_larger_batches(self):
        self.assertEqual(self.train_fixture()[2], {
            "optimizer_steps": 4, "epochs_completed": 2, "selected_epoch": 2})

    def test_counts_accumulated_gradient_steps(self):
        self.assertEqual(self.train_fixture(accumulation=1)[2]["optimizer_steps"], 3)

    def test_counts_completed_epochs_separately_from_selected_epoch(self):
        self.assertEqual(self.train_fixture(early_stop=True)[2], {
            "optimizer_steps": 4, "epochs_completed": 2, "selected_epoch": 1})

    def test_default_trainer_return_contract_preserved(self):
        self.assertEqual(len(self.train_fixture(stats=False)), 2)


if __name__ == "__main__":
    unittest.main()
