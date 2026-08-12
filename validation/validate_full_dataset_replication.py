#!/usr/bin/env python3
"""Independently validate a full_dataset_replication_v1.0.0 result directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, wilcoxon

MODELS = ("A_raw_mlp", "B_fourier_mlp", "C_fourier_jumps")
COMPARISONS = {
    "B_fourier_mlp_vs_A_raw_mlp": ("B_fourier_mlp", "A_raw_mlp"),
    "C_fourier_jumps_vs_A_raw_mlp": ("C_fourier_jumps", "A_raw_mlp"),
    "C_fourier_jumps_vs_B_fourier_mlp": ("C_fourier_jumps", "B_fourier_mlp"),
}


def summarize(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"n_runs": n, "mean": mean, "sd": sd, "ci95_lower": mean - half, "ci95_upper": mean + half}


def paired(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = left - right
    if np.allclose(delta, 0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(left, right, alternative="two-sided", method="auto")
    return {"mean_accuracy_difference": float(delta.mean()), "wilcoxon_statistic": float(statistic), "two_sided_p_value": float(p_value)}


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir
    report = json.loads((result_dir / "summary.json").read_text())
    with (result_dir / "per_seed_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures: list[str] = []
    seeds = report["configuration"]["seeds"]
    expected_rows = len(seeds) * len(MODELS)
    if len(rows) != expected_rows:
        failures.append(f"Expected {expected_rows} rows, found {len(rows)}")
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in MODELS}
    seen = set()
    for row in rows:
        model, seed = row["model"], int(row["seed"])
        if model not in MODELS:
            failures.append(f"Unexpected model {model}")
            continue
        if (model, seed) in seen:
            failures.append(f"Duplicate record for {model}, seed {seed}")
        seen.add((model, seed))
        grouped[model].append(row)
        for key in ("accuracy", "macro_f1", "runtime_seconds", "parameter_count", "epochs_trained", "best_validation_loss"):
            if not np.isfinite(float(row[key])):
                failures.append(f"Non-finite {key} for {model}, seed {seed}")
        accuracy = float(row["accuracy"])
        f1 = float(row["macro_f1"])
        if not 0.0 <= accuracy <= 1.0 or not 0.0 <= f1 <= 1.0:
            failures.append(f"Out-of-range metric for {model}, seed {seed}")
        prediction_path = result_dir / "predictions" / f"seed_{seed}_{model}.npz"
        if not prediction_path.exists():
            failures.append(f"Missing prediction file {prediction_path.name}")
        else:
            with np.load(prediction_path) as values:
                labels, predictions = values["labels"], values["predictions"]
                if labels.shape != predictions.shape or len(labels) == 0:
                    failures.append(f"Malformed predictions in {prediction_path.name}")
                elif not close(float(np.mean(labels == predictions)), accuracy, tolerance=1e-12):
                    failures.append(f"Prediction accuracy mismatch in {prediction_path.name}")
    recomputed: dict[str, np.ndarray] = {}
    for model in MODELS:
        model_rows = sorted(grouped[model], key=lambda row: int(row["seed"]))
        model_seeds = [int(row["seed"]) for row in model_rows]
        if model_seeds != list(seeds):
            failures.append(f"Seed mismatch for {model}: {model_seeds}")
        accuracy = np.asarray([float(row["accuracy"]) for row in model_rows])
        recomputed[model] = accuracy
        for metric in ("accuracy", "macro_f1", "runtime_seconds"):
            values = np.asarray([float(row[metric]) for row in model_rows])
            expected = report["summary"][model][metric]
            for key, value in summarize(values).items():
                if not close(float(expected[key]), value):
                    failures.append(f"Summary mismatch: {model}/{metric}/{key}")
    for name, (left_name, right_name) in COMPARISONS.items():
        expected = report["paired_tests"][name]
        for key, value in paired(recomputed[left_name], recomputed[right_name]).items():
            if not close(float(expected[key]), value):
                failures.append(f"Paired-test mismatch: {name}/{key}")
    feature_manifest = result_dir / "feature_manifest.json"
    if not feature_manifest.exists():
        failures.append("Missing feature_manifest.json")
    outcome = {
        "status": "pass" if not failures else "fail",
        "protocol": report.get("protocol"),
        "n_metric_rows": len(rows),
        "n_prediction_files_expected": expected_rows,
        "checks": {
            "record_count": len(rows) == expected_rows,
            "unique_seed_model_records": len(seen) == expected_rows,
            "prediction_provenance": not any("prediction" in item for item in failures),
            "summary_recomputation": not any("Summary mismatch" in item for item in failures),
            "paired_test_recomputation": not any("Paired-test mismatch" in item for item in failures),
            "feature_manifest": feature_manifest.exists(),
        },
        "failures": failures,
    }
    (result_dir / "validation_report.json").write_text(json.dumps(outcome, indent=2) + "\n")
    lines = ["# Full Dataset Replication Validation Report", "", f"**Status:** {outcome['status'].upper()}", "", "| Check | Result |", "|---|---|"]
    for key, value in outcome["checks"].items():
        lines.append(f"| {key.replace('_', ' ')} | {'Pass' if value else 'Fail'} |")
    lines.extend(["", f"Metric rows: {len(rows)}; expected prediction files: {expected_rows}."])
    if failures:
        lines.extend(["", "## Failures", "", *[f"- {failure}" for failure in failures]])
    (result_dir / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(outcome, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
