"""Evaluate Fourier/jump representations on a local UCR-format benchmark dataset.

This runner preserves the archive's official train/test partition. It splits only
the training partition into stratified training and validation subsets, repeats
model initialization/training across seeds, and reports test accuracy and macro
F1. It does not download datasets automatically; the researcher must obtain and
cite the data from the UCR/UEA archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import load_ucr_univariate, stratified_train_validation_split
from fourier import signals_to_fourier_features, signals_to_fourier_with_jumps
from models import (
    Conv1DSignalClassifier,
    SignalClassifier,
    SignalClassifierWithJumps,
    evaluate_model,
    prepare_dataloader,
    prepare_dataloader_with_jumps,
    train_model,
)


def set_all_seeds(seed: int) -> None:
    """Set deterministic pseudorandom seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(confusion: np.ndarray) -> float:
    """Compute unweighted class-wise F1 from a square confusion matrix."""
    scores = []
    for index in range(confusion.shape[0]):
        tp = float(confusion[index, index])
        fp = float(np.sum(confusion[:, index]) - tp)
        fn = float(np.sum(confusion[index, :]) - tp)
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def summarize(values: List[float]) -> Dict[str, float]:
    """Return mean, sample SD, and normal-approximation 95% interval."""
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if n > 1 else 0.0
    half_width = float(1.96 * sd / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n_runs": n,
        "mean": mean,
        "sd": sd,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def add_result(rows: List[Dict[str, float]], seed: int, model_name: str, accuracy: float, confusion: np.ndarray) -> None:
    """Append common benchmark metrics for one trained model."""
    rows.append(
        {
            "seed": seed,
            "model": model_name,
            "accuracy": float(accuracy),
            "macro_f1": macro_f1(confusion),
        }
    )


def run_one_seed(
    seed: int,
    dataset: Dict[str, object],
    validation_ratio: float,
    n_modes: int,
    max_jumps: int,
    sigma_type: str,
    jump_feature_mode: str,
    epochs: int,
    batch_size: int,
    device: str,
) -> List[Dict[str, float]]:
    """Train all available representation models for one random seed."""
    set_all_seeds(seed)
    split = stratified_train_validation_split(
        dataset["X_train"], dataset["y_train"], validation_ratio=validation_ratio, seed=seed
    )
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    n_classes = int(dataset["n_classes"])
    series_length = int(dataset["series_length"])
    x = np.linspace(-np.pi, np.pi, series_length, endpoint=False)
    rows: List[Dict[str, float]] = []

    # Model A: raw MLP.
    set_all_seeds(seed + 100)
    model_a = SignalClassifier(input_dim=series_length, n_classes=n_classes)
    train_model(
        model_a,
        prepare_dataloader(split["X_train"], split["y_train"], batch_size),
        prepare_dataloader(split["X_val"], split["y_val"], batch_size, False),
        n_epochs=epochs,
        device=device,
    )
    acc_a, cm_a = evaluate_model(
        model_a, prepare_dataloader(X_test, y_test, batch_size, False), device=device
    )
    add_result(rows, seed, "A_raw_mlp", acc_a, cm_a)

    # Model D: compact CNN on raw samples.
    set_all_seeds(seed + 150)
    model_d = Conv1DSignalClassifier(n_classes=n_classes)
    train_model(
        model_d,
        prepare_dataloader(split["X_train"], split["y_train"], batch_size),
        prepare_dataloader(split["X_val"], split["y_val"], batch_size, False),
        n_epochs=epochs,
        device=device,
    )
    acc_d, cm_d = evaluate_model(
        model_d, prepare_dataloader(X_test, y_test, batch_size, False), device=device
    )
    add_result(rows, seed, "D_cnn", acc_d, cm_d)

    # Model B: Fourier MLP.
    X_train_f = signals_to_fourier_features(split["X_train"], n_modes)
    X_val_f = signals_to_fourier_features(split["X_val"], n_modes)
    X_test_f = signals_to_fourier_features(X_test, n_modes)
    set_all_seeds(seed + 200)
    model_b = SignalClassifier(input_dim=2 * n_modes, n_classes=n_classes)
    train_model(
        model_b,
        prepare_dataloader(X_train_f, split["y_train"], batch_size),
        prepare_dataloader(X_val_f, split["y_val"], batch_size, False),
        n_epochs=epochs,
        device=device,
    )
    acc_b, cm_b = evaluate_model(
        model_b, prepare_dataloader(X_test_f, y_test, batch_size, False), device=device
    )
    add_result(rows, seed, "B_fourier_mlp", acc_b, cm_b)

    # Model C: Fourier MLP plus inferred jump descriptors.
    X_train_fj = signals_to_fourier_with_jumps(
        split["X_train"], x, n_modes=n_modes, sigma_type=sigma_type,
        max_jumps=max_jumps, feature_mode=jump_feature_mode
    )
    X_val_fj = signals_to_fourier_with_jumps(
        split["X_val"], x, n_modes=n_modes, sigma_type=sigma_type,
        max_jumps=max_jumps, feature_mode=jump_feature_mode
    )
    X_test_fj = signals_to_fourier_with_jumps(
        X_test, x, n_modes=n_modes, sigma_type=sigma_type,
        max_jumps=max_jumps, feature_mode=jump_feature_mode
    )
    fourier_dim = 2 * n_modes
    jump_dim = X_train_fj.shape[1] - fourier_dim
    set_all_seeds(seed + 300)
    model_c = SignalClassifierWithJumps(
        fourier_dim=fourier_dim, jump_dim=jump_dim, n_classes=n_classes
    )
    train_model(
        model_c,
        prepare_dataloader_with_jumps(
            X_train_fj[:, :fourier_dim], X_train_fj[:, fourier_dim:], split["y_train"], batch_size
        ),
        prepare_dataloader_with_jumps(
            X_val_fj[:, :fourier_dim], X_val_fj[:, fourier_dim:], split["y_val"], batch_size, False
        ),
        n_epochs=epochs,
        device=device,
        model_type="C",
    )
    acc_c, cm_c = evaluate_model(
        model_c,
        prepare_dataloader_with_jumps(
            X_test_fj[:, :fourier_dim], X_test_fj[:, fourier_dim:], y_test, batch_size, False
        ),
        device=device,
        model_type="C",
    )
    add_result(rows, seed, "C_fourier_jumps", acc_c, cm_c)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help="UCR dataset name, e.g. ECG200")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--max-jumps", type=int, default=4)
    parser.add_argument("--sigma-type", choices=["trig", "poly", "exp"], default="trig")
    parser.add_argument("--jump-feature-mode", choices=["locations", "magnitudes", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test_results" / "ucr")
    args = parser.parse_args()

    dataset = load_ucr_univariate(
        args.data_dir, args.dataset, normalize_per_series=not args.no_normalize
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Dataset: {args.dataset}; train: {len(dataset['y_train'])}; test: {len(dataset['y_test'])}; "
        f"length: {dataset['series_length']}; classes: {dataset['n_classes']}; device: {device}"
    )

    rows: List[Dict[str, float]] = []
    for seed in args.seeds:
        print(f"\n=== Benchmark seed {seed} ===")
        seed_rows = run_one_seed(
            seed=seed,
            dataset=dataset,
            validation_ratio=args.validation_ratio,
            n_modes=args.modes,
            max_jumps=args.max_jumps,
            sigma_type=args.sigma_type,
            jump_feature_mode=args.jump_feature_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        rows.extend(seed_rows)
        for row in seed_rows:
            print(f"{row['model']}: accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}")

    models = ["A_raw_mlp", "B_fourier_mlp", "C_fourier_jumps", "D_cnn"]
    summary = {
        model: {
            "accuracy": summarize([row["accuracy"] for row in rows if row["model"] == model]),
            "macro_f1": summarize([row["macro_f1"] for row in rows if row["model"] == model]),
        }
        for model in models
    }
    grouped_accuracy = {
        model: np.asarray([row["accuracy"] for row in rows if row["model"] == model]) for model in models
    }
    paired_tests = {}
    for left_name, right_name in [("C_fourier_jumps", "A_raw_mlp"), ("C_fourier_jumps", "B_fourier_mlp"), ("C_fourier_jumps", "D_cnn")]:
        left, right = grouped_accuracy[left_name], grouped_accuracy[right_name]
        delta = left - right
        if np.allclose(delta, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(left, right, alternative="two-sided", method="auto")
        paired_tests[f"{left_name}_vs_{right_name}"] = {
            "mean_accuracy_difference": float(np.mean(delta)),
            "wilcoxon_statistic": float(statistic),
            "two_sided_p_value": float(p_value),
        }

    csv_path = args.output_dir / f"{args.dataset}_per_seed_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "model", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": {key: value for key, value in dataset.items() if key not in {"X_train", "y_train", "X_test", "y_test"}},
        "configuration": {
            "seeds": args.seeds,
            "validation_ratio": args.validation_ratio,
            "n_modes": args.modes,
            "max_jumps": args.max_jumps,
            "sigma_type": args.sigma_type,
            "jump_feature_mode": args.jump_feature_mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
        },
        "summary": summary,
        "paired_tests": paired_tests,
        "interpretation": (
            "This result applies only to the named UCR-format dataset and the documented protocol. "
            "It must not be generalized to other signal domains without further evidence."
        ),
    }
    json_path = args.output_dir / f"{args.dataset}_summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
