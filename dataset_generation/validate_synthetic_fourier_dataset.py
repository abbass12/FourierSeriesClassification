#!/usr/bin/env python3
"""Validate Synthetic Fourier Signal Dataset v1.0.0 artifacts.

The validator checks every shard for schema, finite values, label/SNR balance,
metadata counts, stored checksums, exact per-signal SNR calibration, and sampled
agreement between the stored Fourier features and independently recomputed DFTs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_KEYS = {
    "signals", "clean_signals", "labels", "fourier_real", "fourier_imag", "snr_db", "support_left", "support_right"
}
EXPECTED_SPLIT_COUNTS = {"train": 4000, "val": 1000, "test": 1000}
EXPECTED_CLASSES = {"sine": 0, "box": 1, "sawtooth": 2, "exponential": 3, "gaussian": 4}
EXPECTED_SNRS = {"clean": -1.0, "snr30": 30.0, "snr20": 20.0, "snr10": 10.0, "snr0": 0.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_checksums(root: Path) -> int:
    count = 0
    for line in (root / "checksums.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        actual = sha256(root / rel)
        assert_true(digest == actual, f"Checksum mismatch: {rel}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    report_dir = args.report_dir.resolve() if args.report_dir else root
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text())
    schema: dict[str, Any] = json.loads((root / "schema.json").read_text())
    assert_true(manifest["version"] == "synthetic_fourier_noise_v1.0.0", "Unexpected dataset version")
    assert_true(manifest["n_points"] == 1024 and manifest["n_fourier_coefficients"] == 64, "Unexpected dimensions")
    assert_true(schema["arrays"]["signals"]["shape"] == ["n_samples", 1024], "Unexpected signal schema")
    assert_true(len(manifest["shards"]) == 75, f"Expected 75 shards, got {len(manifest['shards'])}")

    total = 0
    by_stratum: Counter[tuple[str, str, str]] = Counter()
    exact_snr_errors: list[float] = []
    max_fft_error = 0.0
    all_sample_ids: set[str] = set()
    for record in manifest["shards"]:
        path = root / record["file"]
        assert_true(path.exists(), f"Missing shard: {path}")
        assert_true(sha256(path) == record["sha256"], f"Manifest checksum mismatch: {path.name}")
        with np.load(path) as shard:
            keys = set(shard.files)
            assert_true(keys == EXPECTED_KEYS, f"Bad schema keys in {path.name}: {keys}")
            signals = shard["signals"]
            clean = shard["clean_signals"]
            labels = shard["labels"]
            fourier = shard["fourier_real"] + 1j * shard["fourier_imag"]
            snr = shard["snr_db"]
            n = signals.shape[0]
            assert_true(signals.dtype == np.float32 and clean.dtype == np.float32, f"Signal dtype error in {path.name}")
            assert_true(labels.dtype == np.int64, f"Label dtype error in {path.name}")
            assert_true(signals.shape == (record["n_samples"], 1024), f"Signal shape error in {path.name}")
            assert_true(clean.shape == signals.shape, f"Clean shape error in {path.name}")
            assert_true(fourier.shape == (n, 64), f"Fourier shape error in {path.name}")
            assert_true(np.isfinite(signals).all() and np.isfinite(clean).all() and np.isfinite(fourier).all(), f"Nonfinite data in {path.name}")
            assert_true(np.all(labels == record["label"]), f"Inconsistent labels in {path.name}")
            expected_snr = -1.0 if record["snr_db"] is None else float(record["snr_db"])
            assert_true(np.allclose(snr, expected_snr), f"Inconsistent SNR labels in {path.name}")
            assert_true(np.all(shard["support_left"] < shard["support_right"]), f"Invalid support boundaries in {path.name}")

            if expected_snr >= 0:
                signal_power = np.mean(clean.astype(np.float64) ** 2, axis=1)
                noise_power = np.mean((signals.astype(np.float64) - clean.astype(np.float64)) ** 2, axis=1)
                realized = 10.0 * np.log10(signal_power / noise_power)
                exact_snr_errors.extend(np.abs(realized - expected_snr).tolist())
            else:
                assert_true(np.array_equal(signals, clean), f"Clean condition contains noise in {path.name}")

            probe = min(8, n)
            recomputed = np.fft.rfft(signals[:probe].astype(np.float64), axis=1, norm="forward")[:, :64]
            max_fft_error = max(max_fft_error, float(np.max(np.abs(recomputed - fourier[:probe]))))
            total += n
            by_stratum[(record["split"], record["class_name"], record["snr_condition"])] += n

    assert_true(total == 150000, f"Expected 150000 signals, got {total}")
    for split, expected_n in EXPECTED_SPLIT_COUNTS.items():
        for class_name in EXPECTED_CLASSES:
            for snr_name in EXPECTED_SNRS:
                actual = by_stratum[(split, class_name, snr_name)]
                assert_true(actual == expected_n, f"Imbalanced stratum {split}/{class_name}/{snr_name}: {actual}")

    metadata_counts = {}
    for split, expected_per_stratum in EXPECTED_SPLIT_COUNTS.items():
        metadata_path = root / "metadata" / f"{split}_metadata.csv"
        with metadata_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_total = expected_per_stratum * len(EXPECTED_CLASSES) * len(EXPECTED_SNRS)
        assert_true(len(rows) == expected_total, f"Metadata row count mismatch for {split}")
        for row in rows:
            sample_id = row["sample_id"]
            assert_true(sample_id not in all_sample_ids, f"Duplicate sample ID: {sample_id}")
            all_sample_ids.add(sample_id)
        metadata_counts[split] = len(rows)

    checksum_count = validate_checksums(root)
    assert_true(max_fft_error < 2e-7, f"Stored DFTs do not match signals, error {max_fft_error}")
    max_snr_error = max(exact_snr_errors, default=0.0)
    assert_true(max_snr_error < 1e-4, f"SNR calibration error too large: {max_snr_error}")

    report = {
        "status": "pass",
        "dataset_version": manifest["version"],
        "total_signals": total,
        "n_shards": len(manifest["shards"]),
        "metadata_rows": metadata_counts,
        "checksum_files_verified": checksum_count,
        "max_absolute_snr_error_db": max_snr_error,
        "max_absolute_fourier_error": max_fft_error,
        "class_snr_split_balance": "exact",
    }
    (report_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (report_dir / "validation_report.md").write_text(
        "# Dataset Validation Report\n\n"
        "| Check | Result |\n|---|---|\n"
        f"| Dataset version | `{report['dataset_version']}` |\n"
        f"| Generated signals | {total:,} |\n"
        f"| Shards validated | {report['n_shards']} |\n"
        f"| Split/class/SNR balance | Exact |\n"
        f"| Metadata rows | {sum(metadata_counts.values()):,} |\n"
        f"| Checksums verified | {checksum_count} |\n"
        f"| Maximum SNR calibration error | {max_snr_error:.3e} dB |\n"
        f"| Maximum sampled DFT error | {max_fft_error:.3e} |\n"
        "\nAll schema, finiteness, boundary, label, balance, metadata, checksum, SNR, and sampled DFT checks passed.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
