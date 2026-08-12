#!/usr/bin/env python3
"""Generate the Synthetic Fourier Signal Dataset v1.0.0.

The generator creates a balanced five-family, five-noise-condition benchmark as
sharded NPZ files. It is designed for CUDA-capable Colab runtimes but supports a
CPU fallback. Every sample includes noisy and clean signals, normalized DFT
coefficients, class labels, noise metadata, and known compact-support boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

VERSION = "synthetic_fourier_noise_v1.0.0"
CLASS_NAMES = ("sine", "box", "sawtooth", "exponential", "gaussian")
SNR_CONDITIONS: tuple[tuple[str, float | None], ...] = (
    ("clean", None), ("snr30", 30.0), ("snr20", 20.0), ("snr10", 10.0), ("snr0", 0.0),
)
SPLIT_COUNTS = {"train": 4000, "val": 1000, "test": 1000}
ROOT_SEED = 20260812
N_POINTS = 1024
N_FOURIER = 64


@dataclass(frozen=True)
class Stratum:
    split: str
    class_name: str
    class_label: int
    snr_name: str
    snr_db: float | None
    n_samples: int
    seed: int


def stable_seed(*parts: object) -> int:
    """Produce a platform-independent 63-bit seed from structured identifiers."""
    encoded = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def make_grid(device: torch.device) -> torch.Tensor:
    return torch.linspace(-math.pi, math.pi, N_POINTS + 1, device=device, dtype=torch.float32)[:-1]


def uniform(generator: torch.Generator, low: float, high: float, n: int, device: torch.device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, generator=generator, device=device)


def normalize_max_abs(values: torch.Tensor) -> torch.Tensor:
    return values / torch.amax(torch.abs(values), dim=1, keepdim=True).clamp_min(1e-8)


def generate_clean_signals(stratum: Stratum, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vectorize one signal-family stratum and return clean values plus parameters."""
    n = stratum.n_samples
    gen = torch.Generator(device=device).manual_seed(stratum.seed)
    x = make_grid(device).unsqueeze(0)
    center = uniform(gen, -0.60, 0.60, n, device).unsqueeze(1)
    half_width = uniform(gen, math.pi / 6, math.pi / 2, n, device).unsqueeze(1)
    relative = x - center
    mask = torch.abs(relative) < half_width
    amplitude = uniform(gen, 0.50, 1.50, n, device).unsqueeze(1)
    p1 = torch.zeros((n, 1), device=device)
    p2 = torch.zeros((n, 1), device=device)

    if stratum.class_name == "sine":
        frequency = uniform(gen, 1.0, 8.0, n, device).unsqueeze(1)
        phase = uniform(gen, -math.pi, math.pi, n, device).unsqueeze(1)
        clean = torch.where(mask, amplitude * torch.sin(frequency * relative + phase), torch.zeros_like(relative))
        p1, p2 = frequency, phase
    elif stratum.class_name == "box":
        clean = torch.where(mask, amplitude, torch.zeros_like(relative))
    elif stratum.class_name == "sawtooth":
        slope = uniform(gen, 0.50, 2.50, n, device).unsqueeze(1)
        clean = torch.where(mask, -amplitude * slope * relative / half_width, torch.zeros_like(relative))
        p1 = slope
    elif stratum.class_name == "exponential":
        decay = uniform(gen, 0.50, 5.00, n, device).unsqueeze(1)
        left_relative = (relative + half_width) / (2.0 * half_width)
        clean = torch.where(mask, amplitude * torch.exp(-decay * left_relative), torch.zeros_like(relative))
        p1 = decay
    elif stratum.class_name == "gaussian":
        shape = uniform(gen, 1.50, 8.00, n, device).unsqueeze(1)
        clean = torch.where(mask, amplitude * torch.exp(-shape * (relative / half_width) ** 2), torch.zeros_like(relative))
        p1 = shape
    else:
        raise ValueError(f"Unknown class: {stratum.class_name}")

    clean = normalize_max_abs(clean)
    params = {
        "center": center.squeeze(1),
        "half_width": half_width.squeeze(1),
        "support_left": (center - half_width).squeeze(1),
        "support_right": (center + half_width).squeeze(1),
        "amplitude": amplitude.squeeze(1),
        "parameter_1": p1.squeeze(1),
        "parameter_2": p2.squeeze(1),
    }
    return clean, params


def add_exact_snr_noise(clean: torch.Tensor, snr_db: float | None, seed: int) -> torch.Tensor:
    """Add noise scaled independently per signal to the requested energy SNR."""
    if snr_db is None:
        return clean.clone()
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    signal_power = torch.mean(clean.square(), dim=1, keepdim=True).clamp_min(1e-12)
    raw_noise_power = torch.mean(noise.square(), dim=1, keepdim=True).clamp_min(1e-12)
    desired_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return clean + noise * torch.sqrt(desired_noise_power / raw_noise_power)


def generate_stratum(stratum: Stratum, device: torch.device) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    clean, params = generate_clean_signals(stratum, device)
    noise_seed = stable_seed(ROOT_SEED, stratum.seed, "noise")
    signals = add_exact_snr_noise(clean, stratum.snr_db, noise_seed)
    coefficients = torch.fft.rfft(signals, n=N_POINTS, dim=1, norm="forward")[:, :N_FOURIER]
    labels = torch.full((stratum.n_samples,), stratum.class_label, device=device, dtype=torch.int64)
    snr_value = -1.0 if stratum.snr_db is None else float(stratum.snr_db)

    arrays = {
        "signals": signals.cpu().numpy().astype(np.float32, copy=False),
        "clean_signals": clean.cpu().numpy().astype(np.float32, copy=False),
        "labels": labels.cpu().numpy(),
        "fourier_real": coefficients.real.cpu().numpy().astype(np.float32, copy=False),
        "fourier_imag": coefficients.imag.cpu().numpy().astype(np.float32, copy=False),
        "snr_db": np.full(stratum.n_samples, snr_value, dtype=np.float32),
        "support_left": params["support_left"].cpu().numpy().astype(np.float32, copy=False),
        "support_right": params["support_right"].cpu().numpy().astype(np.float32, copy=False),
    }
    rows: list[dict[str, object]] = []
    parameter_arrays = {name: values.cpu().numpy() for name, values in params.items()}
    for index in range(stratum.n_samples):
        rows.append({
            "sample_id": f"{stratum.split}_{stratum.class_name}_{stratum.snr_name}_{index:05d}",
            "split": stratum.split,
            "label": stratum.class_label,
            "class_name": stratum.class_name,
            "snr_condition": stratum.snr_name,
            "snr_db": "clean" if stratum.snr_db is None else stratum.snr_db,
            "stratum_seed": stratum.seed,
            "noise_seed": noise_seed,
            **{name: float(values[index]) for name, values in parameter_arrays.items()},
        })
    return arrays, rows


def all_strata() -> Iterator[Stratum]:
    for split, n_samples in SPLIT_COUNTS.items():
        for class_label, class_name in enumerate(CLASS_NAMES):
            for snr_name, snr_db in SNR_CONDITIONS:
                seed = stable_seed(ROOT_SEED, VERSION, split, class_name, snr_name)
                yield Stratum(split, class_name, class_label, snr_name, snr_db, n_samples, seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "sample_id", "split", "label", "class_name", "snr_condition", "snr_db", "stratum_seed", "noise_seed",
        "center", "half_width", "support_left", "support_right", "amplitude", "parameter_1", "parameter_2",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    metadata_dir = output_dir / "metadata"
    shards_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    metadata_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in SPLIT_COUNTS}
    shard_records: list[dict[str, object]] = []
    for index, stratum in enumerate(all_strata(), start=1):
        arrays, rows = generate_stratum(stratum, device)
        shard_name = f"{stratum.split}_{stratum.class_name}_{stratum.snr_name}_000.npz"
        shard_path = shards_dir / shard_name
        np.savez_compressed(shard_path, **arrays)
        metadata_by_split[stratum.split].extend(rows)
        shard_records.append({
            "file": f"shards/{shard_name}",
            "sha256": sha256(shard_path),
            "n_samples": stratum.n_samples,
            "split": stratum.split,
            "class_name": stratum.class_name,
            "label": stratum.class_label,
            "snr_condition": stratum.snr_name,
            "snr_db": stratum.snr_db,
            "stratum_seed": stratum.seed,
        })
        print(f"[{index:02d}/75] wrote {shard_name} ({stratum.n_samples} signals)", flush=True)

    metadata_records: list[dict[str, object]] = []
    for split, rows in metadata_by_split.items():
        path = metadata_dir / f"{split}_metadata.csv"
        write_metadata(path, rows)
        metadata_records.append({"file": f"metadata/{path.name}", "sha256": sha256(path), "n_rows": len(rows)})

    schema = {
        "version": VERSION,
        "arrays": {
            "signals": {"dtype": "float32", "shape": ["n_samples", N_POINTS], "description": "Noisy classifier input."},
            "clean_signals": {"dtype": "float32", "shape": ["n_samples", N_POINTS], "description": "Noiseless latent signal."},
            "labels": {"dtype": "int64", "shape": ["n_samples"], "description": "Signal-family class label."},
            "fourier_real": {"dtype": "float32", "shape": ["n_samples", N_FOURIER], "description": "Real component of normalized noisy-signal DFT."},
            "fourier_imag": {"dtype": "float32", "shape": ["n_samples", N_FOURIER], "description": "Imaginary component of normalized noisy-signal DFT."},
            "snr_db": {"dtype": "float32", "shape": ["n_samples"], "description": "-1 denotes clean; otherwise target noise SNR in dB."},
            "support_left": {"dtype": "float32", "shape": ["n_samples"], "description": "Known left compact-support boundary."},
            "support_right": {"dtype": "float32", "shape": ["n_samples"], "description": "Known right compact-support boundary."},
        },
    }
    schema_path = output_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")

    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root_seed": ROOT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version,
        "platform": platform.platform(),
        "n_points": N_POINTS,
        "n_fourier_coefficients": N_FOURIER,
        "class_names": list(CLASS_NAMES),
        "snr_conditions": [{"name": name, "snr_db": snr} for name, snr in SNR_CONDITIONS],
        "split_counts_per_class_snr": SPLIT_COUNTS,
        "n_total_samples": sum(SPLIT_COUNTS.values()) * len(CLASS_NAMES) * len(SNR_CONDITIONS),
        "schema_file": "schema.json",
        "schema_sha256": sha256(schema_path),
        "shards": shard_records,
        "metadata": metadata_records,
        "generator": "dataset_generation/generate_synthetic_fourier_dataset.py",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_path = output_dir / "checksums.sha256"
    paths = sorted([*shards_dir.glob("*.npz"), *metadata_dir.glob("*.csv"), schema_path, manifest_path])
    checksum_path.write_text("".join(f"{sha256(path)}  {path.relative_to(output_dir)}\n" for path in paths))
    print(f"Completed {VERSION}: {manifest['n_total_samples']} signals on {device}.")


if __name__ == "__main__":
    main()
