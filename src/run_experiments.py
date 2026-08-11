"""
Main Experiment Runner
======================
Runs all three models (A, B, C) and generates results.
Designed to run on Google Colab with free GPU or locally on CPU.
"""

import numpy as np
import torch
import time
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signals import (generate_dataset, generate_grid,
                     train_test_split_signals, SIGNAL_NAMES)
from fourier import (signals_to_fourier_features,
                     signals_to_fourier_with_jumps)
from models import (SignalClassifier, SignalClassifierWithJumps,
                    train_model, evaluate_model,
                    prepare_dataloader, prepare_dataloader_with_jumps)


def run_all_experiments(n_samples: int = 1000,
                        n_points: int = 1500,
                        n_modes_list: list = [10, 20, 30, 50, 75, 100],
                        snr_levels: list = [None, 30, 25, 20, 15],
                        n_epochs: int = 50,
                        batch_size: int = 32,
                        seed: int = 42,
                        results_dir: str = '../results') -> Dict:
    """
    Run complete experiment suite.

    Experiments:
    1. Model A: Raw signal classification at various SNR levels
    2. Model B: Fourier coefficient classification at various N modes
    3. Model C: Fourier + jump features at various N modes
    4. Comparison across SNR levels
    """
    os.makedirs(results_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = generate_grid(n_points)
    results = {}

    # =========================================================
    # Experiment 1: Model A - Raw Signal Classification
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Model A - Raw Signal Data")
    print("=" * 60)

    model_a_results = {}

    for snr in snr_levels:
        snr_label = f"SNR_{snr}dB" if snr is not None else "clean"
        print(f"\n--- {snr_label} ---")

        X, y = generate_dataset(n_samples, n_points, snr_db=snr, seed=seed)
        splits = train_test_split_signals(X, y, seed=seed)

        train_loader = prepare_dataloader(
            splits['X_train'], splits['y_train'], batch_size)
        val_loader = prepare_dataloader(
            splits['X_val'], splits['y_val'], batch_size, shuffle=False)
        test_loader = prepare_dataloader(
            splits['X_test'], splits['y_test'], batch_size, shuffle=False)

        model_a = SignalClassifier(input_dim=n_points, n_classes=5)
        history = train_model(model_a, train_loader, val_loader,
                              n_epochs=n_epochs, device=device)
        accuracy, cm = evaluate_model(model_a, test_loader, device=device)

        model_a_results[snr_label] = {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'history': {k: [float(v) for v in vals]
                        for k, vals in history.items()},
        }
        print(f"  Test Accuracy: {accuracy:.4f}")

    results['model_a'] = model_a_results

    # =========================================================
    # Experiment 2: Model B - Fourier Coefficients
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Model B - Fourier Coefficients")
    print("=" * 60)

    model_b_results = {}

    # Generate clean dataset once
    X_clean, y_clean = generate_dataset(n_samples, n_points, snr_db=None,
                                        seed=seed)
    splits_clean = train_test_split_signals(X_clean, y_clean, seed=seed)

    for n_modes in n_modes_list:
        print(f"\n--- N = {n_modes} modes ---")

        # Convert to Fourier features
        X_train_f = signals_to_fourier_features(splits_clean['X_train'],
                                                n_modes)
        X_val_f = signals_to_fourier_features(splits_clean['X_val'], n_modes)
        X_test_f = signals_to_fourier_features(splits_clean['X_test'], n_modes)

        train_loader = prepare_dataloader(
            X_train_f, splits_clean['y_train'], batch_size)
        val_loader = prepare_dataloader(
            X_val_f, splits_clean['y_val'], batch_size, shuffle=False)
        test_loader = prepare_dataloader(
            X_test_f, splits_clean['y_test'], batch_size, shuffle=False)

        model_b = SignalClassifier(input_dim=2 * n_modes, n_classes=5)
        history = train_model(model_b, train_loader, val_loader,
                              n_epochs=n_epochs, device=device)
        accuracy, cm = evaluate_model(model_b, test_loader, device=device)

        model_b_results[f"N_{n_modes}"] = {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'history': {k: [float(v) for v in vals]
                        for k, vals in history.items()},
        }
        print(f"  Test Accuracy: {accuracy:.4f}")

    results['model_b'] = model_b_results

    # =========================================================
    # Experiment 3: Model C - Fourier + Jump Features
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Model C - Fourier + Jump Features")
    print("=" * 60)

    model_c_results = {}
    max_jumps = 4

    for n_modes in n_modes_list:
        print(f"\n--- N = {n_modes} modes (with jumps) ---")

        # Convert to Fourier + jump features
        X_train_fj = signals_to_fourier_with_jumps(
            splits_clean['X_train'], x, n_modes, max_jumps=max_jumps)
        X_val_fj = signals_to_fourier_with_jumps(
            splits_clean['X_val'], x, n_modes, max_jumps=max_jumps)
        X_test_fj = signals_to_fourier_with_jumps(
            splits_clean['X_test'], x, n_modes, max_jumps=max_jumps)

        fourier_dim = 2 * n_modes
        jump_dim = 2 * max_jumps

        # Split features for Model C's two-branch architecture
        train_loader = prepare_dataloader_with_jumps(
            X_train_fj[:, :fourier_dim], X_train_fj[:, fourier_dim:],
            splits_clean['y_train'], batch_size)
        val_loader = prepare_dataloader_with_jumps(
            X_val_fj[:, :fourier_dim], X_val_fj[:, fourier_dim:],
            splits_clean['y_val'], batch_size, shuffle=False)
        test_loader = prepare_dataloader_with_jumps(
            X_test_fj[:, :fourier_dim], X_test_fj[:, fourier_dim:],
            splits_clean['y_test'], batch_size, shuffle=False)

        model_c = SignalClassifierWithJumps(
            fourier_dim=fourier_dim, jump_dim=jump_dim, n_classes=5)
        history = train_model(model_c, train_loader, val_loader,
                              n_epochs=n_epochs, device=device,
                              model_type='C')
        accuracy, cm = evaluate_model(model_c, test_loader, device=device,
                                      model_type='C')

        model_c_results[f"N_{n_modes}"] = {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'history': {k: [float(v) for v in vals]
                        for k, vals in history.items()},
        }
        print(f"  Test Accuracy: {accuracy:.4f}")

    results['model_c'] = model_c_results

    # =========================================================
    # Experiment 4: SNR Comparison (all models at N=50)
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: SNR Comparison (N=50)")
    print("=" * 60)

    n_modes_compare = 50
    snr_comparison = {}

    for snr in snr_levels:
        snr_label = f"SNR_{snr}dB" if snr is not None else "clean"
        print(f"\n--- {snr_label} ---")

        X, y = generate_dataset(n_samples, n_points, snr_db=snr, seed=seed)
        splits = train_test_split_signals(X, y, seed=seed)

        # Model B
        X_train_f = signals_to_fourier_features(splits['X_train'],
                                                n_modes_compare)
        X_val_f = signals_to_fourier_features(splits['X_val'],
                                              n_modes_compare)
        X_test_f = signals_to_fourier_features(splits['X_test'],
                                               n_modes_compare)

        train_loader_b = prepare_dataloader(
            X_train_f, splits['y_train'], batch_size)
        val_loader_b = prepare_dataloader(
            X_val_f, splits['y_val'], batch_size, shuffle=False)
        test_loader_b = prepare_dataloader(
            X_test_f, splits['y_test'], batch_size, shuffle=False)

        model_b = SignalClassifier(input_dim=2 * n_modes_compare, n_classes=5)
        train_model(model_b, train_loader_b, val_loader_b,
                    n_epochs=n_epochs, device=device)
        acc_b, _ = evaluate_model(model_b, test_loader_b, device=device)

        # Model C
        X_train_fj = signals_to_fourier_with_jumps(
            splits['X_train'], x, n_modes_compare, max_jumps=max_jumps)
        X_val_fj = signals_to_fourier_with_jumps(
            splits['X_val'], x, n_modes_compare, max_jumps=max_jumps)
        X_test_fj = signals_to_fourier_with_jumps(
            splits['X_test'], x, n_modes_compare, max_jumps=max_jumps)

        fourier_dim = 2 * n_modes_compare
        jump_dim = 2 * max_jumps

        train_loader_c = prepare_dataloader_with_jumps(
            X_train_fj[:, :fourier_dim], X_train_fj[:, fourier_dim:],
            splits['y_train'], batch_size)
        val_loader_c = prepare_dataloader_with_jumps(
            X_val_fj[:, :fourier_dim], X_val_fj[:, fourier_dim:],
            splits['y_val'], batch_size, shuffle=False)
        test_loader_c = prepare_dataloader_with_jumps(
            X_test_fj[:, :fourier_dim], X_test_fj[:, fourier_dim:],
            splits['y_test'], batch_size, shuffle=False)

        model_c = SignalClassifierWithJumps(
            fourier_dim=fourier_dim, jump_dim=jump_dim, n_classes=5)
        train_model(model_c, train_loader_c, val_loader_c,
                    n_epochs=n_epochs, device=device, model_type='C')
        acc_c, _ = evaluate_model(model_c, test_loader_c, device=device,
                                  model_type='C')

        snr_comparison[snr_label] = {
            'model_b_accuracy': float(acc_b),
            'model_c_accuracy': float(acc_c),
        }
        print(f"  Model B: {acc_b:.4f}, Model C: {acc_c:.4f}")

    results['snr_comparison'] = snr_comparison

    # Save results
    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


# Type annotation fix for the function
from typing import Dict


if __name__ == '__main__':
    results = run_all_experiments(
        n_samples=1000,
        n_points=1500,
        n_modes_list=[10, 20, 30, 50, 75, 100],
        snr_levels=[None, 30, 25, 20, 15],
        n_epochs=50,
        batch_size=32,
        seed=42,
    )
