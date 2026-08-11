"""Run all experiments for the Fourier Signal Classification paper."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signals import generate_dataset, generate_grid, train_test_split_signals
from fourier import signals_to_fourier_features, signals_to_fourier_with_jumps
from models import (SignalClassifier, SignalClassifierWithJumps,
                    train_model, evaluate_model,
                    prepare_dataloader, prepare_dataloader_with_jumps)
import numpy as np
import torch
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

n_samples = 500
n_points = 1500
x = generate_grid(n_points)
seed = 42
batch_size = 32
n_epochs = 40
results = {}

# === Model A: Raw signals ===
print('\n=== MODEL A: Raw Signal Data ===')
snr_levels = [None, 25, 20, 15]
model_a_results = {}
for snr in snr_levels:
    label = f'SNR_{snr}dB' if snr else 'clean'
    X, y = generate_dataset(n_samples, n_points, snr_db=snr, seed=seed)
    splits = train_test_split_signals(X, y, seed=seed)
    tl = prepare_dataloader(splits['X_train'], splits['y_train'], batch_size)
    vl = prepare_dataloader(splits['X_val'], splits['y_val'], batch_size, False)
    tel = prepare_dataloader(splits['X_test'], splits['y_test'], batch_size, False)
    model = SignalClassifier(n_points, 5)
    hist = train_model(model, tl, vl, n_epochs=n_epochs, device=device)
    acc, cm = evaluate_model(model, tel, device=device)
    model_a_results[label] = {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist(),
        'history': {k: [float(v) for v in vals] for k, vals in hist.items()}
    }
    print(f'  {label}: {acc:.4f}')
results['model_a'] = model_a_results

# === Model B: Fourier coefficients ===
print('\n=== MODEL B: Fourier Coefficients ===')
X_clean, y_clean = generate_dataset(n_samples, n_points, snr_db=None, seed=seed)
splits_clean = train_test_split_signals(X_clean, y_clean, seed=seed)
model_b_results = {}
for n_modes in [10, 20, 30, 50, 75, 100]:
    X_tr = signals_to_fourier_features(splits_clean['X_train'], n_modes)
    X_va = signals_to_fourier_features(splits_clean['X_val'], n_modes)
    X_te = signals_to_fourier_features(splits_clean['X_test'], n_modes)
    tl = prepare_dataloader(X_tr, splits_clean['y_train'], batch_size)
    vl = prepare_dataloader(X_va, splits_clean['y_val'], batch_size, False)
    tel = prepare_dataloader(X_te, splits_clean['y_test'], batch_size, False)
    model = SignalClassifier(2 * n_modes, 5)
    hist = train_model(model, tl, vl, n_epochs=n_epochs, device=device)
    acc, cm = evaluate_model(model, tel, device=device)
    model_b_results[f'N_{n_modes}'] = {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist(),
        'history': {k: [float(v) for v in vals] for k, vals in hist.items()}
    }
    print(f'  N={n_modes}: {acc:.4f}')
results['model_b'] = model_b_results

# === Model C: Fourier + Jumps ===
print('\n=== MODEL C: Fourier + Jump Features ===')
model_c_results = {}
max_jumps = 4
for n_modes in [10, 20, 30, 50, 75, 100]:
    X_tr = signals_to_fourier_with_jumps(splits_clean['X_train'], x, n_modes, max_jumps=max_jumps)
    X_va = signals_to_fourier_with_jumps(splits_clean['X_val'], x, n_modes, max_jumps=max_jumps)
    X_te = signals_to_fourier_with_jumps(splits_clean['X_test'], x, n_modes, max_jumps=max_jumps)
    fd = 2 * n_modes
    jd = 2 * max_jumps
    tl = prepare_dataloader_with_jumps(X_tr[:, :fd], X_tr[:, fd:], splits_clean['y_train'], batch_size)
    vl = prepare_dataloader_with_jumps(X_va[:, :fd], X_va[:, fd:], splits_clean['y_val'], batch_size, False)
    tel = prepare_dataloader_with_jumps(X_te[:, :fd], X_te[:, fd:], splits_clean['y_test'], batch_size, False)
    model = SignalClassifierWithJumps(fd, jd, 5)
    hist = train_model(model, tl, vl, n_epochs=n_epochs, device=device, model_type='C')
    acc, cm = evaluate_model(model, tel, device=device, model_type='C')
    model_c_results[f'N_{n_modes}'] = {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist(),
        'history': {k: [float(v) for v in vals] for k, vals in hist.items()}
    }
    print(f'  N={n_modes}: {acc:.4f}')
results['model_c'] = model_c_results

# === Save ===
os.makedirs('results', exist_ok=True)
with open('results/experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f"Model A (clean): {results['model_a']['clean']['accuracy']:.4f}")
print(f"Model B (N=50):  {results['model_b']['N_50']['accuracy']:.4f}")
print(f"Model C (N=50):  {results['model_c']['N_50']['accuracy']:.4f}")
print('\nDone!')
