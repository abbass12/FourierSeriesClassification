"""Repeated-seed validation for Fourier Signal Classification.

This script evaluates the raw-signal baseline (A), Fourier features (B), and
Fourier-plus-jump features (C) over independent dataset/model seeds. It writes
sample-level results, aggregate confidence intervals, and paired Wilcoxon tests.

The default configuration is deliberately modest for a laptop. For a submission
run, use the Full_Experiment_Colab notebook or invoke this script with a larger
sample count and at least 10 seeds on a GPU-enabled environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from signals import generate_dataset, generate_grid, train_test_split_signals
from fourier import signals_to_fourier_features, signals_to_fourier_with_jumps
from models import (
    SignalClassifier,
    Conv1DSignalClassifier,
    SignalClassifierWithJumps,
    evaluate_model,
    prepare_dataloader,
    prepare_dataloader_with_jumps,
    train_model,
)


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def summarize(values: List[float]) -> Dict[str, float]:
    """Return mean, sample SD, and normal-approximation 95% CI."""
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if n > 1 else 0.0
    half_width = float(1.96 * sd / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n_runs": n,
        "mean_accuracy": mean,
        "sd_accuracy": sd,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def run_one_seed(
    seed: int,
    n_samples_per_type: int,
    n_points: int,
    n_modes: int,
    snr_db: float | None,
    epochs: int,
    batch_size: int,
    device: str,
) -> List[Dict[str, float]]:
    """Train A, B, and C once each and return accuracy records."""
    set_all_seeds(seed)
    X, y = generate_dataset(
        n_samples_per_type=n_samples_per_type,
        n_points=n_points,
        snr_db=snr_db,
        seed=seed,
    )
    split = train_test_split_signals(X, y, seed=seed)
    x = generate_grid(n_points)
    rows: List[Dict[str, float]] = []

    # Model A: raw samples.
    set_all_seeds(seed + 100)
    a_train = prepare_dataloader(split["X_train"], split["y_train"], batch_size)
    a_val = prepare_dataloader(split["X_val"], split["y_val"], batch_size, False)
    a_test = prepare_dataloader(split["X_test"], split["y_test"], batch_size, False)
    model_a = SignalClassifier(input_dim=n_points, n_classes=5)
    train_model(model_a, a_train, a_val, n_epochs=epochs, device=device)
    acc_a, _ = evaluate_model(model_a, a_test, device=device)
    rows.append({"seed": seed, "model": "A_raw", "accuracy": float(acc_a)})

    # Model D: compact one-dimensional CNN baseline on raw samples.
    set_all_seeds(seed + 150)
    d_train = prepare_dataloader(split["X_train"], split["y_train"], batch_size)
    d_val = prepare_dataloader(split["X_val"], split["y_val"], batch_size, False)
    d_test = prepare_dataloader(split["X_test"], split["y_test"], batch_size, False)
    model_d = Conv1DSignalClassifier(n_classes=5)
    train_model(model_d, d_train, d_val, n_epochs=epochs, device=device)
    acc_d, _ = evaluate_model(model_d, d_test, device=device)
    rows.append({"seed": seed, "model": "D_cnn", "accuracy": float(acc_d)})

    # Model B: real and imaginary Fourier coefficients.
    X_train_f = signals_to_fourier_features(split["X_train"], n_modes)
    X_val_f = signals_to_fourier_features(split["X_val"], n_modes)
    X_test_f = signals_to_fourier_features(split["X_test"], n_modes)
    set_all_seeds(seed + 200)
    b_train = prepare_dataloader(X_train_f, split["y_train"], batch_size)
    b_val = prepare_dataloader(X_val_f, split["y_val"], batch_size, False)
    b_test = prepare_dataloader(X_test_f, split["y_test"], batch_size, False)
    model_b = SignalClassifier(input_dim=2 * n_modes, n_classes=5)
    train_model(model_b, b_train, b_val, n_epochs=epochs, device=device)
    acc_b, _ = evaluate_model(model_b, b_test, device=device)
    rows.append({"seed": seed, "model": "B_fourier", "accuracy": float(acc_b)})

    # Model C: Fourier coefficients plus fixed-length jump features.
    max_jumps = 4
    X_train_fj = signals_to_fourier_with_jumps(
        split["X_train"], x, n_modes=n_modes, max_jumps=max_jumps
    )
    X_val_fj = signals_to_fourier_with_jumps(
        split["X_val"], x, n_modes=n_modes, max_jumps=max_jumps
    )
    X_test_fj = signals_to_fourier_with_jumps(
        split["X_test"], x, n_modes=n_modes, max_jumps=max_jumps
    )
    fourier_dim = 2 * n_modes
    set_all_seeds(seed + 300)
    c_train = prepare_dataloader_with_jumps(
        X_train_fj[:, :fourier_dim], X_train_fj[:, fourier_dim:], split["y_train"], batch_size
    )
    c_val = prepare_dataloader_with_jumps(
        X_val_fj[:, :fourier_dim], X_val_fj[:, fourier_dim:], split["y_val"], batch_size, False
    )
    c_test = prepare_dataloader_with_jumps(
        X_test_fj[:, :fourier_dim], X_test_fj[:, fourier_dim:], split["y_test"], batch_size, False
    )
    model_c = SignalClassifierWithJumps(
        fourier_dim=fourier_dim, jump_dim=2 * max_jumps, n_classes=5
    )
    train_model(model_c, c_train, c_val, n_epochs=epochs, device=device, model_type="C")
    acc_c, _ = evaluate_model(model_c, c_test, device=device, model_type="C")
    rows.append({"seed": seed, "model": "C_fourier_jumps", "accuracy": float(acc_c)})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--samples-per-type", type=int, default=300)
    parser.add_argument("--points", type=int, default=1500)
    parser.add_argument("--modes", type=int, default=50)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test_results" / "repeated_seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}; samples/type: {args.samples_per_type}; modes: {args.modes}")

    rows: List[Dict[str, float]] = []
    for seed in args.seeds:
        print(f"\n=== Validation seed {seed} ===")
        seed_rows = run_one_seed(
            seed=seed,
            n_samples_per_type=args.samples_per_type,
            n_points=args.points,
            n_modes=args.modes,
            snr_db=args.snr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        rows.extend(seed_rows)
        for row in seed_rows:
            print(f"{row['model']}: {row['accuracy']:.4f}")

    csv_path = args.output_dir / "per_seed_accuracy.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "model", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    grouped = {model: [row["accuracy"] for row in rows if row["model"] == model]
               for model in ["A_raw", "B_fourier", "C_fourier_jumps", "D_cnn"]}
    summary = {model: summarize(values) for model, values in grouped.items()}

    a = np.asarray(grouped["A_raw"])
    b = np.asarray(grouped["B_fourier"])
    c = np.asarray(grouped["C_fourier_jumps"])
    d = np.asarray(grouped["D_cnn"])
    tests = {}
    for left, right, label in [
        (c, a, "C_vs_A"), (c, b, "C_vs_B"), (c, d, "C_vs_D"),
        (b, a, "B_vs_A"), (b, d, "B_vs_D"), (a, d, "A_vs_D"),
    ]:
        delta = left - right
        if np.allclose(delta, 0):
            p_value = 1.0
            statistic = 0.0
        else:
            statistic, p_value = wilcoxon(left, right, alternative="two-sided", method="auto")
        tests[label] = {
            "mean_accuracy_difference": float(np.mean(delta)),
            "wilcoxon_statistic": float(statistic),
            "two_sided_p_value": float(p_value),
        }

    report = {
        "configuration": {
            "seeds": args.seeds,
            "n_samples_per_type": args.samples_per_type,
            "n_points": args.points,
            "n_modes": args.modes,
            "snr_db": args.snr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
        },
        "summary": summary,
        "paired_tests": tests,
        "interpretation": (
            "These results characterize only the specified synthetic generator and protocol. "
            "They do not establish generalization to real-world signal-classification tasks."
        ),
    }
    json_path = args.output_dir / "validation_summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
