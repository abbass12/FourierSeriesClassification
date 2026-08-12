#!/usr/bin/env python3
"""Regenerate every empirical table in paper/main.tex under protocol v2.2.2.

The runner writes only to the user-selected output root. It leaves archived outputs
untouched. A separate validator recomputes metrics from the emitted CSV files and
checks the fresh artifacts against the frozen manuscript protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "manuscript_validation_protocol_v2.2.2.json"
VALIDATOR = ROOT / "validation" / "validate_manuscript_results.py"


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ucr_command(data_dir: Path, output_dir: Path, sigma_type: str, feature_mode: str) -> list[str]:
    return [
        sys.executable, "run_ucr_benchmark.py",
        "--data-dir", str(data_dir),
        "--dataset", "ECG200",
        "--seeds", "11", "23", "37",
        "--validation-ratio", "0.2",
        "--modes", "32",
        "--max-jumps", "4",
        "--sigma-type", sigma_type,
        "--jump-feature-mode", feature_mode,
        "--epochs", "50",
        "--batch-size", "16",
        "--output-dir", str(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()

    commands = [
        {
            "name": "synthetic_smoke",
            "command": [
                sys.executable, "run_repeated_validation.py",
                "--seeds", "11", "23", "37",
                "--samples-per-type", "80",
                "--points", "1500",
                "--modes", "50",
                "--epochs", "12",
                "--batch-size", "32",
                "--output-dir", str(output_root / "repeated_seed_smoke_v2"),
            ],
        },
        {
            "name": "ecg200_screening",
            "command": ucr_command(data_dir, output_root / "ECG200_screening", "trig", "both"),
        },
        {
            "name": "ecg200_locations",
            "command": ucr_command(data_dir, output_root / "ECG200_locations", "trig", "locations"),
        },
        {
            "name": "ecg200_magnitudes",
            "command": ucr_command(data_dir, output_root / "ECG200_magnitudes", "trig", "magnitudes"),
        },
        {
            "name": "ecg200_poly",
            "command": ucr_command(data_dir, output_root / "ECG200_poly", "poly", "both"),
        },
        {
            "name": "ecg200_exp",
            "command": ucr_command(data_dir, output_root / "ECG200_exp", "exp", "both"),
        },
    ]

    manifest = {
        "protocol_version": protocol["protocol_version"],
        "output_root": str(output_root),
        "commands": commands,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for item in commands:
        run(item["command"], ROOT)

    if not args.skip_validation:
        run(
            [
                sys.executable, str(VALIDATOR),
                "--artifact-root", str(output_root),
                "--output-dir", str(output_root / "validation_report"),
            ],
            ROOT,
        )

    print(f"Completed protocol {protocol['protocol_version']}: {output_root}")


if __name__ == "__main__":
    main()
