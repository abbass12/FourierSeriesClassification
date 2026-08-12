#!/usr/bin/env python3
"""Validate every empirical manuscript result against versioned CSV/JSON artifacts.

This validator checks three layers:
1. stored JSON configuration and reported summary values against the frozen protocol;
2. means and sample SDs recomputed directly from per-seed CSV results;
3. displayed manuscript table values against the expected rounded values.

It validates reproducibility artifacts, not scientific generalizability. Fresh reruns are
validated by pointing --root to their output directory after matching its expected layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "manuscript_validation_protocol_v2.2.2.json"
PAPER_PATH = ROOT / "paper" / "main.tex"

TOL = 1e-12


class ValidationError(Exception):
    """Raised when a validation condition is not met."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def assert_close(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise ValidationError(f"{label}: expected {expected!r}, got {actual!r}")


def sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_synthetic(result_dir: Path, expected: dict[str, Any]) -> dict[str, Any]:
    summary_path = result_dir / "validation_summary.json"
    csv_path = result_dir / "per_seed_accuracy.csv"
    summary = load_json(summary_path)
    rows = csv_rows(csv_path)
    config = summary["configuration"]
    expected_config = {
        "seeds": [11, 23, 37],
        "n_samples_per_type": 80,
        "n_points": 1500,
        "n_modes": 50,
        "snr_db": None,
        "epochs": 12,
        "batch_size": 32,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValidationError(f"synthetic config {key}: expected {value!r}, got {config.get(key)!r}")

    expected_summary = expected["expected_summary"]
    observed: dict[str, Any] = {}
    for model, values in expected_summary.items():
        records = [float(row["accuracy"]) for row in rows if row["model"] == model]
        if len(records) != 3:
            raise ValidationError(f"synthetic {model}: expected 3 per-seed rows, got {len(records)}")
        recomputed_mean = float(np.mean(records))
        recomputed_sd = sample_sd(records)
        stored = summary["summary"][model]
        assert_close(f"synthetic {model} CSV mean", recomputed_mean, stored["mean_accuracy"])
        assert_close(f"synthetic {model} CSV SD", recomputed_sd, stored["sd_accuracy"])
        assert_close(f"synthetic {model} expected mean", stored["mean_accuracy"], values["mean_accuracy"])
        assert_close(f"synthetic {model} expected SD", stored["sd_accuracy"], values["sd_accuracy"])
        observed[model] = {"mean_accuracy": recomputed_mean, "sd_accuracy": recomputed_sd}

    for label, value in expected["expected_paired_p_values"].items():
        assert_close(f"synthetic {label} p-value", summary["paired_tests"][label]["two_sided_p_value"], value)

    return {
        "status": "pass",
        "kind": "synthetic_smoke",
        "summary_path": str(summary_path.relative_to(ROOT)),
        "csv_path": str(csv_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path),
        "csv_sha256": sha256(csv_path),
        "observed": observed,
    }


def validate_ecg(
    name: str,
    result_dir: Path,
    expected_config: dict[str, Any],
    expected_models: dict[str, Any],
) -> dict[str, Any]:
    summary_path = result_dir / "ECG200_summary.json"
    csv_path = result_dir / "ECG200_per_seed_metrics.csv"
    summary = load_json(summary_path)
    rows = csv_rows(csv_path)
    config = summary["configuration"]
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValidationError(f"{name} config {key}: expected {value!r}, got {config.get(key)!r}")

    observed: dict[str, Any] = {}
    for model, expected_values in expected_models.items():
        records = [row for row in rows if row["model"] == model]
        if len(records) != 3:
            raise ValidationError(f"{name} {model}: expected 3 per-seed rows, got {len(records)}")
        accuracies = [float(row["accuracy"]) for row in records]
        f1s = [float(row["macro_f1"]) for row in records]
        stored = summary["summary"][model]
        assert_close(f"{name} {model} CSV accuracy mean", float(np.mean(accuracies)), stored["accuracy"]["mean"])
        assert_close(f"{name} {model} CSV accuracy SD", sample_sd(accuracies), stored["accuracy"]["sd"])
        assert_close(f"{name} {model} CSV F1 mean", float(np.mean(f1s)), stored["macro_f1"]["mean"])
        assert_close(f"{name} {model} CSV F1 SD", sample_sd(f1s), stored["macro_f1"]["sd"])

        # The primary screen contains four model entries. Ablations only constrain the changed C model.
        if "accuracy_mean" in expected_values:
            assert_close(f"{name} {model} expected accuracy mean", stored["accuracy"]["mean"], expected_values["accuracy_mean"])
            assert_close(f"{name} {model} expected accuracy SD", stored["accuracy"]["sd"], expected_values["accuracy_sd"])
            assert_close(f"{name} {model} expected F1 mean", stored["macro_f1"]["mean"], expected_values["macro_f1_mean"])
            assert_close(f"{name} {model} expected F1 SD", stored["macro_f1"]["sd"], expected_values["macro_f1_sd"])
        observed[model] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_sd": sample_sd(accuracies),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_sd": sample_sd(f1s),
        }

    return {
        "status": "pass",
        "kind": name,
        "summary_path": str(summary_path.relative_to(ROOT)),
        "csv_path": str(csv_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path),
        "csv_sha256": sha256(csv_path),
        "observed": observed,
    }


def validate_paper_tokens(protocol: dict[str, Any]) -> dict[str, Any]:
    text = PAPER_PATH.read_text()
    expected_tokens = [
        "87.92\\%", "80.42\\%", "89.33\\%", "91.67\\%", "79.00\\%", "85.00\\%",
        "84.33\\%", "83.33\\%", "Trigonometric", "Polynomial", "Exponential",
    ]
    missing = [token for token in expected_tokens if token not in text]
    if missing:
        raise ValidationError(f"manuscript is missing expected reported table tokens: {missing}")
    return {
        "status": "pass",
        "kind": "manuscript_displayed_values",
        "paper_path": str(PAPER_PATH.relative_to(ROOT)),
        "paper_sha256": sha256(PAPER_PATH),
        "checked_tokens": expected_tokens,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Manuscript Result Validation Report", "",
        f"Protocol: `{report['protocol_version']}`", "",
        "| Validation target | Status | Primary artifact |", "|---|---|---|",
    ]
    for item in report["checks"]:
        artifact = item.get("summary_path", item.get("paper_path", ""))
        lines.append(f"| {item['kind']} | {item['status']} | `{artifact}` |")
    lines.extend([
        "",
        "## Scope", "",
        "All checks confirm internal artifact consistency, configuration agreement, recomputed summary statistics, and displayed manuscript values. They do not establish external generalization or confirmatory statistical power.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation" / "reports")
    parser.add_argument(
        "--artifact-root", type=Path, default=ROOT / "test_results",
        help="Directory containing result folders with the standard manuscript-validation names.",
    )
    args = parser.parse_args()
    protocol = load_json(PROTOCOL_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = args.artifact_root.resolve()

    checks: list[dict[str, Any]] = []
    checks.append(validate_synthetic(artifact_root / "repeated_seed_smoke_v2", protocol["synthetic_smoke"]))

    base = protocol["ecg200_screening"]
    expected_config = {
        "seeds": [11, 23, 37],
        "validation_ratio": 0.2,
        "n_modes": 32,
        "max_jumps": 4,
        "epochs": 50,
        "batch_size": 16,
    }
    primary_models = {
        model: {
            "accuracy_mean": values["accuracy_mean"],
            "accuracy_sd": values["accuracy_sd"],
            "macro_f1_mean": values["macro_f1_mean"],
            "macro_f1_sd": values["macro_f1_sd"],
        }
        for model, values in base["expected_summary"].items()
    }
    checks.append(validate_ecg(
        "ecg200_screening",
        artifact_root / "ECG200_screening",
        {**expected_config, "sigma_type": "trig", "jump_feature_mode": "both"},
        primary_models,
    ))

    ablations = [
        ("ecg200_jump_location_ablation", "ECG200_locations", {"sigma_type": "trig", "jump_feature_mode": "locations"}),
        ("ecg200_jump_magnitude_ablation", "ECG200_magnitudes", {"sigma_type": "trig", "jump_feature_mode": "magnitudes"}),
        ("ecg200_sigma_polynomial_ablation", "ECG200_poly", {"sigma_type": "poly", "jump_feature_mode": "both"}),
        ("ecg200_sigma_exponential_ablation", "ECG200_exp", {"sigma_type": "exp", "jump_feature_mode": "both"}),
    ]
    for protocol_key, directory, overrides in ablations:
        checks.append(validate_ecg(
            protocol_key,
            artifact_root / directory,
            {**expected_config, **overrides},
            protocol[protocol_key]["expected"],
        ))

    checks.append(validate_paper_tokens(protocol))
    report = {"protocol_version": protocol["protocol_version"], "checks": checks}
    json_path = args.output_dir / "manuscript_result_validation.json"
    md_path = args.output_dir / "manuscript_result_validation.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(markdown_report(report))
    print(f"PASS: {len(checks)} validation checks")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
