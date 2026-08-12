import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_generation.generate_synthetic_fourier_dataset import (
    CLASS_NAMES,
    Stratum,
    add_exact_snr_noise,
    generate_stratum,
    stable_seed,
)


def test_stable_seed_is_repeatable_and_distinguishes_strata():
    assert stable_seed("train", "sine", "snr20") == stable_seed("train", "sine", "snr20")
    assert stable_seed("train", "sine", "snr20") != stable_seed("test", "sine", "snr20")


def test_all_families_generate_expected_shapes_and_metadata():
    device = torch.device("cpu")
    for label, class_name in enumerate(CLASS_NAMES):
        stratum = Stratum("train", class_name, label, "snr20", 20.0, 5, stable_seed(class_name))
        arrays, rows = generate_stratum(stratum, device)
        assert arrays["signals"].shape == (5, 1024)
        assert arrays["clean_signals"].shape == (5, 1024)
        assert arrays["fourier_real"].shape == (5, 64)
        assert arrays["labels"].tolist() == [label] * 5
        assert len(rows) == 5
        assert np.isfinite(arrays["signals"]).all()
        assert np.all(arrays["support_left"] < arrays["support_right"])


def test_noise_has_the_requested_per_signal_snr():
    clean = torch.ones((8, 1024), dtype=torch.float32)
    noisy = add_exact_snr_noise(clean, 10.0, 1234)
    actual = 10.0 * torch.log10(torch.mean(clean.square(), 1) / torch.mean((noisy - clean).square(), 1))
    assert torch.allclose(actual, torch.full_like(actual, 10.0), atol=1e-5)


def test_clean_condition_is_identical_to_latent_signal():
    clean = torch.randn((4, 1024), dtype=torch.float32)
    assert torch.equal(clean, add_exact_snr_noise(clean, None, 5))


def test_stored_fourier_features_match_direct_fft():
    stratum = Stratum("val", "box", 1, "snr30", 30.0, 3, stable_seed("box"))
    arrays, _ = generate_stratum(stratum, torch.device("cpu"))
    actual = np.fft.rfft(arrays["signals"], axis=1, norm="forward")[:, :64]
    stored = arrays["fourier_real"] + 1j * arrays["fourier_imag"]
    assert np.max(np.abs(actual - stored)) < 2e-7
