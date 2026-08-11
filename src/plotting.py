"""
Plotting Module
===============
Generates all figures for the paper and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Dict, Optional, List
import os

from signals import generate_grid, SIGNAL_NAMES, SIGNAL_GENERATORS, add_noise
from fourier import (compute_fourier_coefficients, fourier_partial_sum,
                     generalized_conjugate_partial_sum)


# Style settings for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_signal_types(save_path: Optional[str] = None):
    """Plot all five signal types (Figure 1 in paper)."""
    x = generate_grid(1500)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i, (name, gen_func) in enumerate(SIGNAL_GENERATORS.items()):
        signal = gen_func(x)
        axes[i].plot(x, signal, 'b-', linewidth=1.2)
        axes[i].set_title(name.capitalize())
        axes[i].set_xlabel('x')
        axes[i].set_xlim([-np.pi, np.pi])
        axes[i].set_ylim([-1.2, 1.2])
        axes[i].grid(True, alpha=0.3)

    axes[0].set_ylabel('Amplitude')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_signals_with_noise(snr_db: float = 20,
                            save_path: Optional[str] = None):
    """Plot signals with added noise (Figure 2 in paper)."""
    x = generate_grid(1500)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i, (name, gen_func) in enumerate(SIGNAL_GENERATORS.items()):
        signal = gen_func(x)
        noisy = add_noise(signal, snr_db)
        axes[i].plot(x, noisy, 'b-', linewidth=0.5, alpha=0.8)
        axes[i].set_title(f"{name.capitalize()} (SNR={snr_db}dB)")
        axes[i].set_xlabel('x')
        axes[i].set_xlim([-np.pi, np.pi])
        axes[i].grid(True, alpha=0.3)

    axes[0].set_ylabel('Amplitude')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_signals_and_coefficients(save_path: Optional[str] = None):
    """Plot signals alongside their Fourier coefficients (Figure 3)."""
    x = generate_grid(1500)
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))

    for i, (name, gen_func) in enumerate(SIGNAL_GENERATORS.items()):
        signal = gen_func(x)
        coeffs = compute_fourier_coefficients(signal, n_modes=100)

        # Signal plot
        axes[i, 0].plot(x, signal, 'b-', linewidth=1.2)
        axes[i, 0].set_title(f"{name.capitalize()} Signal")
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_xlim([-np.pi, np.pi])
        axes[i, 0].grid(True, alpha=0.3)

        # Coefficient magnitude plot
        k = np.arange(-50, 50)
        axes[i, 1].semilogy(k, np.abs(coeffs) + 1e-16, 'r.-', markersize=3)
        axes[i, 1].set_title(f"{name.capitalize()} |$\\hat{{f}}_k$|")
        axes[i, 1].set_xlabel('Mode k')
        axes[i, 1].set_ylabel('|Coefficient|')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_fourier_approximations(n_modes_list: List[int] = [5, 20, 50],
                                save_path_prefix: Optional[str] = None):
    """Plot Fourier approximations at different mode counts (Figures 4-6)."""
    x = generate_grid(1500)

    for n_modes in n_modes_list:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))

        for i, (name, gen_func) in enumerate(SIGNAL_GENERATORS.items()):
            signal = gen_func(x)
            coeffs = compute_fourier_coefficients(signal, n_modes)
            reconstruction = fourier_partial_sum(coeffs, x)

            axes[i].plot(x, signal, 'b-', linewidth=1.0, label='Original',
                         alpha=0.5)
            axes[i].plot(x, reconstruction, 'r-', linewidth=1.2,
                         label=f'N={n_modes}')
            axes[i].set_title(name.capitalize())
            axes[i].set_xlabel('x')
            axes[i].set_xlim([-np.pi, np.pi])
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

        axes[0].set_ylabel('Amplitude')
        plt.tight_layout()

        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_N{n_modes}.png")
            plt.close()
        else:
            plt.show()


def plot_edge_detection(save_path: Optional[str] = None):
    """Plot edge detection results using concentration factors (Figure 7)."""
    x = generate_grid(1500)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    sigma_types = ['trig', 'poly', 'exp']
    sigma_labels = ['Trigonometric $\\sigma$', 'Polynomial $\\sigma$',
                    'Exponential $\\sigma$']

    for row, (sigma_type, sigma_label) in enumerate(
            zip(sigma_types, sigma_labels)):
        for col, (name, gen_func) in enumerate(SIGNAL_GENERATORS.items()):
            signal = gen_func(x)
            coeffs = compute_fourier_coefficients(signal, n_modes=100)
            edge_fn = generalized_conjugate_partial_sum(
                coeffs, x, sigma_type=sigma_type)

            axes[row, col].plot(x, edge_fn, 'r-', linewidth=1.0)
            axes[row, col].axhline(y=0, color='k', linewidth=0.5)
            axes[row, col].set_xlim([-np.pi, np.pi])
            axes[row, col].grid(True, alpha=0.3)

            if row == 0:
                axes[row, col].set_title(name.capitalize())
            if col == 0:
                axes[row, col].set_ylabel(sigma_label)
            if row == 2:
                axes[row, col].set_xlabel('x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_history(history: Dict, model_name: str,
                          save_path: Optional[str] = None):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_acc']) + 1)

    # Accuracy
    ax1.plot(epochs, history['train_acc'], 'b-', label='Training')
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history['train_loss'], 'b-', label='Training')
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                          save_path: Optional[str] = None):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SIGNAL_NAMES, yticklabels=SIGNAL_NAMES,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} - Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_accuracy_vs_modes(results: Dict, save_path: Optional[str] = None):
    """Plot accuracy vs number of Fourier modes for Models B and C."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Extract data
    modes_b = []
    acc_b = []
    for key, val in results['model_b'].items():
        n = int(key.split('_')[1])
        modes_b.append(n)
        acc_b.append(val['accuracy'])

    modes_c = []
    acc_c = []
    for key, val in results['model_c'].items():
        n = int(key.split('_')[1])
        modes_c.append(n)
        acc_c.append(val['accuracy'])

    ax.plot(modes_b, acc_b, 'bo-', label='Model B (Fourier only)',
            markersize=8)
    ax.plot(modes_c, acc_c, 'rs-', label='Model C (Fourier + Jumps)',
            markersize=8)

    # Add Model A baseline
    if 'model_a' in results and 'clean' in results['model_a']:
        acc_a = results['model_a']['clean']['accuracy']
        ax.axhline(y=acc_a, color='g', linestyle='--',
                   label=f'Model A (Raw data): {acc_a:.3f}')

    ax.set_xlabel('Number of Fourier Modes (N)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Classification Accuracy vs. Number of Fourier Modes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_snr_comparison(results: Dict, save_path: Optional[str] = None):
    """Plot accuracy vs SNR for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    snr_labels = []
    acc_a = []
    acc_b = []
    acc_c = []

    # Model A results
    for key, val in results['model_a'].items():
        snr_labels.append(key)
        acc_a.append(val['accuracy'])

    # SNR comparison (Models B and C) - use model_b/model_c N=50 if snr_comparison missing
    if 'snr_comparison' in results:
        for key in snr_labels:
            if key in results['snr_comparison']:
                acc_b.append(results['snr_comparison'][key]['model_b_accuracy'])
                acc_c.append(results['snr_comparison'][key]['model_c_accuracy'])
    else:
        # Use available data
        if 'model_b' in results and 'N_50' in results['model_b']:
            acc_b = [results['model_b']['N_50']['accuracy']] * len(snr_labels)
        if 'model_c' in results and 'N_50' in results['model_c']:
            acc_c = [results['model_c']['N_50']['accuracy']] * len(snr_labels)

    x_pos = range(len(snr_labels))
    ax.plot(x_pos, acc_a, 'go-', label='Model A (Raw)', markersize=8)
    ax.plot(x_pos, acc_b, 'bo-', label='Model B (Fourier)', markersize=8)
    ax.plot(x_pos, acc_c, 'rs-', label='Model C (Fourier+Jumps)',
            markersize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(snr_labels, rotation=45)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Classification Accuracy vs. Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_all_paper_figures(results: Dict,
                               output_dir: str = '../results/figures'):
    """Generate all figures needed for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Figure 1: Signal Types...")
    plot_signal_types(os.path.join(output_dir, 'fig1_signals.png'))

    print("Generating Figure 2: Signals with Noise...")
    plot_signals_with_noise(20, os.path.join(output_dir,
                                             'fig2_signals_noise.png'))

    print("Generating Figure 3: Signals and Coefficients...")
    plot_signals_and_coefficients(
        os.path.join(output_dir, 'fig3_signals_coeffs.png'))

    print("Generating Figures 4-6: Fourier Approximations...")
    plot_fourier_approximations(
        [5, 20, 50],
        os.path.join(output_dir, 'fig_fourier_approx'))

    print("Generating Figure 7: Edge Detection...")
    plot_edge_detection(os.path.join(output_dir, 'fig7_edge_detection.png'))

    print("Generating accuracy vs modes plot...")
    plot_accuracy_vs_modes(results,
                           os.path.join(output_dir, 'fig_acc_vs_modes.png'))

    print("Generating SNR comparison plot...")
    plot_snr_comparison(results,
                        os.path.join(output_dir, 'fig_snr_comparison.png'))

    # Training histories and confusion matrices
    if 'model_a' in results and 'clean' in results['model_a']:
        hist_a = results['model_a']['clean']['history']
        plot_training_history(hist_a, 'Model A',
                             os.path.join(output_dir, 'fig_modelA_train.png'))
        cm_a = np.array(results['model_a']['clean']['confusion_matrix'])
        plot_confusion_matrix(cm_a, 'Model A',
                              os.path.join(output_dir, 'fig_modelA_cm.png'))

    if 'model_b' in results and 'N_50' in results['model_b']:
        hist_b = results['model_b']['N_50']['history']
        plot_training_history(hist_b, 'Model B (N=50)',
                             os.path.join(output_dir, 'fig_modelB_train.png'))
        cm_b = np.array(results['model_b']['N_50']['confusion_matrix'])
        plot_confusion_matrix(cm_b, 'Model B (N=50)',
                              os.path.join(output_dir, 'fig_modelB_cm.png'))

    if 'model_c' in results and 'N_50' in results['model_c']:
        hist_c = results['model_c']['N_50']['history']
        plot_training_history(hist_c, 'Model C (N=50)',
                             os.path.join(output_dir, 'fig_modelC_train.png'))
        cm_c = np.array(results['model_c']['N_50']['confusion_matrix'])
        plot_confusion_matrix(cm_c, 'Model C (N=50)',
                              os.path.join(output_dir, 'fig_modelC_cm.png'))

    print(f"\nAll figures saved to {output_dir}")
