"""Bounded CPU readiness checks around the upstream HC-InitNet pipeline."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..algorithms.neural_networks import DEBNetHC
from ..data.load_data import load_col_types, load_dataframes
from ..data.prepare_data_pytorch import LogScaleTensorTransformer, prepare_data_tensors

ROOT = Path(__file__).resolve().parents[2]
MODEL_SETTINGS = ("shared_hidden_layers", "par_hidden_layers", "dropout_prob", "clamp_function", "use_skip_connection")
DATA_FILES = ("init_net.csv", "train.csv", "val.csv", "test.csv", "types_of_col.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frames(frames, full, col_types):
    """Audit identities and semantics before calling original preparation."""
    required = list(dict.fromkeys(sum((col_types[k]["all"] for k in ("input", "output", "mask")), [])))
    seen = set()
    if full.index.has_duplicates or full.index.isna().any():
        raise ValueError("complete dataset has duplicate/missing species IDs")
    for split, df in frames.items():
        if df.empty or df.index.has_duplicates or df.index.isna().any():
            raise ValueError(f"{split}: empty data or duplicate/missing species IDs")
        if seen.intersection(df.index):
            raise ValueError(f"{split}: species overlap between splits")
        seen.update(df.index)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{split}: missing columns {sorted(missing)}")
        for kind in ("input", "output"):
            values = df[col_types[kind]["all"]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{split}: non-finite {kind} values")
            # Even masked output placeholders enter upstream log-scaler fitting.
            if (df[col_types[kind]["log"]].to_numpy(dtype=float) <= 0).any():
                raise ValueError(f"{split}: non-positive log {kind} values")
        for name in col_types["mask"]["all"]:
            if not df[name].isin([True, False]).all():
                raise ValueError(f"{split}: invalid eligibility mask {name}")
        eligible = df[[f"estim_{c}" for c in col_types["output"]["all"]]].to_numpy(dtype=bool)
        if not eligible.any(axis=1).all():
            raise ValueError(f"{split}: row has no eligible output")
        try:
            pd.testing.assert_frame_equal(df[required], full.loc[df.index, required], check_exact=True)
        except (AssertionError, KeyError) as exc:
            raise ValueError(f"{split}: differs from complete dataset") from exc
        if not full.loc[df.index, "data_split"].eq(split).all():
            raise ValueError(f"{split}: complete-table split labels disagree")
    if seen != set(full.index):
        raise ValueError("split union differs from complete dataset")


def prepare(dataset, batch_size):
    raw = {s: pd.read_csv(dataset / f"{s}.csv", index_col=0) for s in ("train", "val", "test")}
    full = pd.read_csv(dataset / "init_net.csv", index_col=0)
    types = load_col_types(dataset.name, datasets_folder=dataset.parent)
    validate_frames(raw, full, types)
    frames, types = load_dataframes(dataset.name, datasets_folder=dataset.parent)
    tensors, loaders, datasets, scalers = prepare_data_tensors(
        frames, types, batch_size, "log_standardize", torch.device("cpu"))
    for split, ds in datasets.items():
        if not all(torch.isfinite(t).all() for t in ds.tensors):
            raise ValueError(f"{split}: non-finite prepared tensors")
        for kind in ("input", "output"):
            x = tensors[split][kind]
            torch.testing.assert_close(scalers[kind].inverse_transform(scalers[kind].transform(x)),
                                       x, rtol=2e-5, atol=2e-5)
    manifest = {"files": {n: {"sha256": sha256(dataset / n), "bytes": (dataset / n).stat().st_size}
                           for n in DATA_FILES},
                "counts": {s: len(df) for s, df in raw.items()},
                "input_columns": types["input"]["all"], "output_columns": types["output"]["all"],
                "mask_columns": types["mask"]["all"],
                "species_ids": {s: df.index.tolist() for s, df in raw.items()},
                "eligible_outputs": {s: int(t["mask"].sum()) for s, t in tensors.items()}}
    return types, tensors, loaders, scalers, manifest


def make_model(training, types, scalers):
    return DEBNetHC(col_types=types, input_transformer=scalers["input"],
                    output_transformer=scalers["output"], **{k: training[k] for k in MODEL_SETTINGS})


def check_scalers(tensors, scalers):
    for kind in ("input", "output"):
        scaler = scalers[kind]
        x = tensors["train"][kind].clone()
        x[:, scaler.log_cols] = torch.log(x[:, scaler.log_cols])
        std, mean = torch.std_mean(x[:, scaler.scale_cols], dim=0)
        torch.testing.assert_close(scaler.mean, mean)
        torch.testing.assert_close(scaler.std, std)


def restore_checkpoint(path, config, types):
    """Reject semantic mismatches and rebuild preprocessing from saved metadata."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "column_types" not in saved:
        raise ValueError("checkpoint lacks full column metadata; regenerate with this runner")
    if saved["config"]["model_class"] != config["model_class"]:
        raise ValueError("checkpoint model_class mismatch")
    for key in MODEL_SETTINGS + ("scaling_type",):
        if saved["config"]["training"][key] != config["training"][key]:
            raise ValueError(f"checkpoint training setting mismatch: {key}")
    if saved["column_types"] != types:
        raise ValueError("checkpoint column order or transformation metadata mismatch")
    if config["model_class"] != "DEBNetHC" or config["training"]["scaling_type"] != "log_standardize":
        raise ValueError("checkpoint requires CPU DEBNetHC with log_standardize")
    scalers = {}
    for kind in ("input", "output"):
        cols = saved["column_types"][kind]
        scalers[kind] = LogScaleTensorTransformer(cols["all"], cols["log"], cols["scale"])
        for key in ("mean", "std"):
            setattr(scalers[kind], key, saved["scalers"][kind][key])
    model = make_model(saved["config"]["training"], saved["column_types"], scalers)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def run(config, mode, output):
    start = time.monotonic()
    if config["device"] != "cpu" or config["model_class"] != "DEBNetHC":
        raise ValueError("readiness runner supports only CPU DEBNetHC")
    if not 1 <= config["epochs"] <= 2 or not 0 < config["timeout_seconds"] <= 600:
        raise ValueError("smoke limits exceed the accepted bounds")
    if not 0 < config["max_output_bytes"] <= 100_000_000:
        raise ValueError("output bound exceeds the accepted limit")
    if config["training"]["scaling_type"] != "log_standardize":
        raise ValueError("readiness runner supports only log_standardize preprocessing")
    if output.exists():
        raise ValueError("output directory already exists; choose a fresh run directory")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    dataset = (ROOT / config["dataset"]).resolve()
    types, tensors, _, scalers, manifest = prepare(dataset, config["training"]["batch_size"])
    manifest["dataset_directory"] = str(dataset)
    check_scalers(tensors, scalers)
    model = make_model(config["training"], types, scalers).eval()
    with torch.no_grad():
        prediction = model(scalers["input"].transform(tensors["val"]["input"]))
        if not torch.isfinite(prediction).all():
            raise ValueError("non-finite validation predictions")
    report = {"mode": mode, "seed": config["seed"], "device": "cpu", "test_scored": False,
              "optimizer_updates": 0, "epochs_completed": 0, "dataset_directory": str(dataset),
              "baseline_commit": config["baseline_commit"],
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "config": config, "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "numpy", "pandas", "scikit-learn", "ray")},
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(__file__), ROOT / "project_code/algorithms/neural_networks.py",
                                 ROOT / "project_code/train/train_pytorch_model.py",
                                 ROOT / "project_code/data/prepare_data_pytorch.py"]}}
    if mode == "smoke":
        from .train_pytorch_model import train_neural_network
        initial = {}
        gradient_checks = []

        def track(grad):
            if not torch.isfinite(grad).all():
                raise ValueError("non-finite training gradient")
            gradient_checks.append(True)
            return grad

        def factory(clamp_function=config["training"]["clamp_function"], **kwargs):
            model = DEBNetHC(clamp_function=clamp_function, **kwargs)
            initial.update({k: v.detach().clone() for k, v in model.named_parameters()})
            for p in model.parameters():
                p.register_hook(track)
            return model

        training = copy.deepcopy(config["training"])
        training["max_epochs"] = config["epochs"]
        model, metrics, stats = train_neural_network(training, factory, dataset.name, device=torch.device("cpu"),
                                              seed=config["seed"], verbose=False,
                                              datasets_folder=dataset.parent, return_training_stats=True)
        if not np.isfinite(metrics.to_numpy(dtype=float)).all() or not gradient_checks:
            raise ValueError("non-finite metrics or no checked gradients")
        changed = any(not torch.equal(initial[k], v.detach()) for k, v in model.named_parameters())
        if not changed:
            raise ValueError("optimizer did not update model parameters")
        model.eval()
        val_x = scalers["input"].transform(tensors["val"]["input"])
        with torch.no_grad():
            expected = model(val_x)
        if not torch.isfinite(expected).all():
            raise ValueError("non-finite post-training predictions")
        report.update({"optimizer_updates": stats["optimizer_steps"],
                       "epochs_completed": stats["epochs_completed"],
                       "selected_epoch": stats["selected_epoch"], "epochs_requested": config["epochs"], "finite_gradient_checks": len(gradient_checks),
                       "weights_changed": changed, "validation_metrics": metrics.to_dict()})
        payload = {"state_dict": model.state_dict(), "config": config,
                   "column_types": types,
                   "scalers": {k: {a: getattr(model.input_transformer if k == "input" else
                                             model.output_transformer, a) for a in ("mean", "std")}
                               for k in ("input", "output")}}
        # Check size before writing, rather than treating the cap as a postcondition.
        import io
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        if buffer.tell() + 1_000_000 > config["max_output_bytes"]:
            raise ValueError("checkpoint would exceed output bound")
        output.mkdir(parents=True)
        checkpoint = output / "checkpoint.pt"
        checkpoint.write_bytes(buffer.getvalue())
        restored = restore_checkpoint(checkpoint, config, types)
        with torch.no_grad():
            actual = restored(val_x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report["reload_predictions_identical"] = True
    else:
        output.mkdir(parents=True)
    report["duration_seconds"] = time.monotonic() - start
    (output / "data-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > config["max_output_bytes"]:
        raise ValueError("output bound exceeded")
    print(json.dumps({"mode": mode, "output": str(output), "duration_seconds": report["duration_seconds"],
                      "bytes": size, "optimizer_updates": report["optimizer_updates"]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hc_initnet.json")
    parser.add_argument("--mode", choices=("dry-run", "smoke"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    def timeout(signum, frame):
        raise TimeoutError("readiness run exceeded its accepted runtime")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config["timeout_seconds"])
    try:
        run(config, args.mode, args.output.resolve())
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
