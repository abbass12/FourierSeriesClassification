#!/usr/bin/env python3
"""Run the frozen full-dataset replication protocol.

This script evaluates the manuscript's three primary representations on the
fixed train/validation/test files of Synthetic Fourier Noise Dataset v1.0.0.
It intentionally treats the 1024-point release as a confirmatory extension,
not a byte-for-byte rerun of the historical 1500-point pilot.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import sici
from scipy.stats import t, wilcoxon
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from models import SignalClassifier, SignalClassifierWithJumps, train_model  # noqa: E402

MODELS = ("A_raw_mlp", "B_fourier_mlp", "C_fourier_jumps")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def load_split_condition(dataset_dir: Path, split: str, condition: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted((dataset_dir / "shards").glob(f"{split}_*_{condition}_000.npz"))
    if len(paths) != 5:
        raise FileNotFoundError(f"Expected five {split}/{condition} shards, found {len(paths)} in {dataset_dir}")
    signals, labels = [], []
    for path in paths:
        with np.load(path) as shard:
            signals.append(np.asarray(shard["signals"], dtype=np.float32))
            labels.append(np.asarray(shard["labels"], dtype=np.int64))
    x = np.concatenate(signals, axis=0)
    y = np.concatenate(labels, axis=0)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError(f"Malformed {split}/{condition} data")
    return x, y


def centered_coefficients(x: np.ndarray, n_modes: int) -> np.ndarray:
    n_points = x.shape[1]
    if n_modes <= 0 or n_modes > n_points:
        raise ValueError("n_modes must lie in 1..n_points")
    values = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1) / n_points
    start = n_points // 2 - n_modes // 2
    return values[:, start:start + n_modes]


def fourier_features(x: np.ndarray, n_modes: int) -> np.ndarray:
    coeffs = centered_coefficients(x, n_modes)
    return np.concatenate([coeffs.real, coeffs.imag], axis=1).astype(np.float32)


def trig_sigma(k: np.ndarray, n_max: int) -> np.ndarray:
    si_pi, _ = sici(np.pi)
    return np.pi * np.sin(np.pi * np.abs(k) / max(n_max, 1)) / si_pi


def jump_features(x: np.ndarray, n_modes: int, max_jumps: int, batch_size: int = 256) -> np.ndarray:
    """Compute the historical top-peak descriptor in batches without test leakage."""
    n_samples, n_points = x.shape
    coeffs = centered_coefficients(x, n_modes)
    half = n_modes // 2
    k = np.arange(-half, half) if n_modes % 2 == 0 else np.arange(-half, half + 1)
    k = k[:n_modes]
    x_grid = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    sigma = trig_sigma(k, max(np.max(np.abs(k)), 1))
    weights = 1j * np.sign(k) * sigma
    basis = np.exp(1j * np.outer(x_grid, k)).astype(np.complex64)
    features = np.zeros((n_samples, 2 * max_jumps), dtype=np.float32)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        response = np.real((coeffs[start:stop] * weights) @ basis.T)
        for local_index, edge in enumerate(response):
            magnitude = np.abs(edge)
            peak = float(np.max(magnitude))
            if peak <= 1e-10:
                continue
            candidates = np.flatnonzero(magnitude / peak > 0.3)
            selections: list[int] = []
            cursor = 0
            while cursor < len(candidates):
                end = cursor
                while end < len(candidates) - 1 and candidates[end + 1] - candidates[end] <= 10:
                    end += 1
                group = candidates[cursor:end + 1]
                selections.append(int(group[np.argmax(magnitude[group])]))
                cursor = end + 1
            selections = sorted(selections, key=lambda idx: -magnitude[idx])[:max_jumps]
            if selections:
                local = start + local_index
                features[local, :len(selections)] = x_grid[selections]
                features[local, max_jumps:max_jumps + len(selections)] = edge[selections]
    return features


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, pin_memory=True)


def make_jump_loader(fourier_x: np.ndarray, jumps_x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(fourier_x, dtype=torch.float32),
        torch.as_tensor(jumps_x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, pin_memory=True)


def evaluate_predictions(model: torch.nn.Module, loader: DataLoader, device: str, model_type: str = "AB") -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval().to(device)
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == "C":
                a, b, y = (item.to(device, non_blocking=True) for item in batch)
                logits = model(a, b)
            else:
                x, y = (item.to(device, non_blocking=True) for item in batch)
                logits = model(x)
            labels_all.append(y.cpu().numpy())
            predictions_all.append(torch.argmax(logits, dim=1).cpu().numpy())
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    n_classes = int(max(labels.max(), predictions.max())) + 1
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    return float(np.mean(labels == predictions)), confusion, labels, predictions


def macro_f1(confusion: np.ndarray) -> float:
    values = []
    for i in range(confusion.shape[0]):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - tp)
        fn = float(confusion[i, :].sum() - tp)
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"n_runs": n, "mean": mean, "sd": sd, "ci95_lower": mean - half, "ci95_upper": mean + half}


def paired(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = left - right
    if np.allclose(delta, 0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(left, right, alternative="two-sided", method="auto")
    return {"mean_accuracy_difference": float(delta.mean()), "wilcoxon_statistic": float(statistic), "two_sided_p_value": float(p_value)}


def train_and_score(name: str, seed: int, data: dict[str, Any], args: argparse.Namespace, device: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    set_seeds(seed + {"A_raw_mlp": 100, "B_fourier_mlp": 200, "C_fourier_jumps": 300}[name])
    n_classes = 5
    start = time.perf_counter()
    if name == "A_raw_mlp":
        model = SignalClassifier(input_dim=data["raw_train"].shape[1], n_classes=n_classes)
        train_loader = make_loader(data["raw_train"], data["y_train"], args.batch_size, True, seed)
        val_loader = make_loader(data["raw_val"], data["y_val"], args.batch_size, False, seed)
        test_loader = make_loader(data["raw_test"], data["y_test"], args.batch_size, False, seed)
        history = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.patience, device, "AB")
        accuracy, confusion, labels, predictions = evaluate_predictions(model, test_loader, device, "AB")
    elif name == "B_fourier_mlp":
        model = SignalClassifier(input_dim=data["fourier_train"].shape[1], n_classes=n_classes)
        train_loader = make_loader(data["fourier_train"], data["y_train"], args.batch_size, True, seed)
        val_loader = make_loader(data["fourier_val"], data["y_val"], args.batch_size, False, seed)
        test_loader = make_loader(data["fourier_test"], data["y_test"], args.batch_size, False, seed)
        history = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.patience, device, "AB")
        accuracy, confusion, labels, predictions = evaluate_predictions(model, test_loader, device, "AB")
    else:
        model = SignalClassifierWithJumps(fourier_dim=data["fourier_train"].shape[1], jump_dim=data["jump_train"].shape[1], n_classes=n_classes)
        train_loader = make_jump_loader(data["fourier_train"], data["jump_train"], data["y_train"], args.batch_size, True, seed)
        val_loader = make_jump_loader(data["fourier_val"], data["jump_val"], data["y_val"], args.batch_size, False, seed)
        test_loader = make_jump_loader(data["fourier_test"], data["jump_test"], data["y_test"], args.batch_size, False, seed)
        history = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.patience, device, "C")
        accuracy, confusion, labels, predictions = evaluate_predictions(model, test_loader, device, "C")
    elapsed = time.perf_counter() - start
    record = {
        "seed": seed, "model": name, "accuracy": accuracy, "macro_f1": macro_f1(confusion),
        "runtime_seconds": elapsed, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_trained": len(history["val_loss"]), "best_validation_loss": float(min(history["val_loss"])),
        "confusion_matrix": confusion.tolist(),
    }
    return record, labels, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", default="clean", choices=["clean", "snr30", "snr20", "snr10", "snr0"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71, 89, 107, 131, 149, 167])
    parser.add_argument("--modes", type=int, default=50)
    parser.add_argument("--max-jumps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    raw_train, y_train = load_split_condition(args.dataset_dir, "train", args.condition)
    raw_val, y_val = load_split_condition(args.dataset_dir, "val", args.condition)
    raw_test, y_test = load_split_condition(args.dataset_dir, "test", args.condition)
    print(f"Loaded {args.condition}: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, points={raw_train.shape[1]}", flush=True)
    print("Computing fixed Fourier and jump descriptors once before all seeds...", flush=True)
    data: dict[str, Any] = {
        "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "fourier_train": fourier_features(raw_train, args.modes),
        "fourier_val": fourier_features(raw_val, args.modes),
        "fourier_test": fourier_features(raw_test, args.modes),
        "jump_train": jump_features(raw_train, args.modes, args.max_jumps),
        "jump_val": jump_features(raw_val, args.modes, args.max_jumps),
        "jump_test": jump_features(raw_test, args.modes, args.max_jumps),
    }
    feature_manifest = {
        "condition": args.condition, "n_modes": args.modes, "max_jumps": args.max_jumps,
        "raw_shapes": {split: list(data[f"raw_{split}"].shape) for split in ("train", "val", "test")},
        "fourier_shapes": {split: list(data[f"fourier_{split}"].shape) for split in ("train", "val", "test")},
        "jump_shapes": {split: list(data[f"jump_{split}"].shape) for split in ("train", "val", "test")},
    }
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2) + "\n")

    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        for model_name in MODELS:
            record, labels, predictions = train_and_score(model_name, seed, data, args, device)
            records.append(record)
            np.savez_compressed(prediction_dir / f"seed_{seed}_{model_name}.npz", labels=labels, predictions=predictions)
            print(f"{model_name}: accuracy={record['accuracy']:.4f}; macro_f1={record['macro_f1']:.4f}; epochs={record['epochs_trained']}; seconds={record['runtime_seconds']:.1f}", flush=True)

    metric_columns = ["seed", "model", "accuracy", "macro_f1", "runtime_seconds", "parameter_count", "epochs_trained", "best_validation_loss", "confusion_matrix"]
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
            writer.writerow(row)

    grouped = {name: [record for record in records if record["model"] == name] for name in MODELS}
    result_summary = {
        name: {
            "accuracy": summary([record["accuracy"] for record in grouped[name]]),
            "macro_f1": summary([record["macro_f1"] for record in grouped[name]]),
            "runtime_seconds": summary([record["runtime_seconds"] for record in grouped[name]]),
            "parameter_count": int(grouped[name][0]["parameter_count"]),
        }
        for name in MODELS
    }
    arrays = {name: np.array([record["accuracy"] for record in grouped[name]], dtype=float) for name in MODELS}
    tests = {
        "B_fourier_mlp_vs_A_raw_mlp": paired(arrays["B_fourier_mlp"], arrays["A_raw_mlp"]),
        "C_fourier_jumps_vs_A_raw_mlp": paired(arrays["C_fourier_jumps"], arrays["A_raw_mlp"]),
        "C_fourier_jumps_vs_B_fourier_mlp": paired(arrays["C_fourier_jumps"], arrays["B_fourier_mlp"]),
    }
    report = {
        "protocol": "full_dataset_replication_v1.0.0",
        "repository_commit": git_commit(),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_condition": args.condition,
        "configuration": {key: getattr(args, key) for key in ("seeds", "modes", "max_jumps", "epochs", "batch_size", "lr", "patience", "device")},
        "hardware": {"torch_version": torch.__version__, "cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "summary": result_summary,
        "paired_tests": tests,
        "replication_scope": "A confirmatory 1024-point dataset extension, not a literal numerical rerun of the historical 1500-point pilot.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
