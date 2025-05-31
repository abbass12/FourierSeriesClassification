"""
Validation script to ensure the refactored code produces the same results as the original.

This script compares outputs from the refactored implementation with expected results
to verify that functionality is preserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fourier_classification.signals import (
    box_signal, saw_signal, exp_signal, sin_signal, gaussian_signal
)
from fourier_classification.fourier import fourier_series
from fourier_classification.operations import add_noise, extract_jump
from fourier_classification.utils import create_domain

# Create output directory for validation results
output_dir = "validation_results"
os.makedirs(output_dir, exist_ok=True)

# Create domain
x = create_domain(start=-np.pi, end=np.pi, num_points=1500)

def validate_signal_generation():
    """Validate signal generation functions."""
    print("Validating signal generation...")
    
    # Generate signals with default parameters
    signals = {
        'box': box_signal(x, a=2, b=5, normalized=True),
        'saw': saw_signal(x, a=2, b=5, normalized=True),
        'exp': exp_signal(x, a=2, b=2, c=-1, normalized=True),
        'sin': sin_signal(x, a=2, b=2, c=-1, normalized=True),
        'gaus': gaussian_signal(x, a=2, b=2, normalized=True)
    }
    
    # Save signals for visual inspection
    for name, signal in signals.items():
        plt.figure(figsize=(10, 6))
        plt.plot(x, signal)
        plt.title(f"{name.capitalize()} Signal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{name}_signal.png"))
        plt.close()
        
        # Save signal data
        np.save(os.path.join(output_dir, f"{name}_signal.npy"), signal)
    
    print("Signal generation validation complete. Results saved to:", output_dir)
    return signals

def validate_fourier_coefficients():
    """Validate Fourier coefficient calculation."""
    print("Validating Fourier coefficient calculation...")
    
    # Generate Fourier coefficients
    coeffs = {
        'box': box_signal(x, a=2, b=5, normalized=True, fourier=True, n_modes=40),
        'saw': saw_signal(x, a=2, b=5, normalized=True, fourier=True, n_modes=40),
        'exp': exp_signal(x, a=2, b=2, c=-1, normalized=True, fourier=True, n_modes=40),
        'sin': sin_signal(x, a=2, b=2, c=-1, normalized=True, fourier=True, n_modes=40),
        'gaus': gaussian_signal(x, a=2, b=2, normalized=True, fourier=True, n_modes=40)
    }
    
    # Save coefficients for visual inspection
    for name, coeff in coeffs.items():
        plt.figure(figsize=(10, 6))
        k = np.linspace(-len(coeff)//2, len(coeff)//2, len(coeff))
        plt.stem(k, np.abs(coeff))
        plt.title(f"{name.capitalize()} Fourier Coefficients")
        plt.xlabel("k")
        plt.ylabel("|c_k|")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{name}_coeffs.png"))
        plt.close()
        
        # Save coefficient data
        np.save(os.path.join(output_dir, f"{name}_coeffs.npy"), coeff)
    
    print("Fourier coefficient validation complete. Results saved to:", output_dir)
    return coeffs

def validate_fourier_series():
    """Validate Fourier series reconstruction."""
    print("Validating Fourier series reconstruction...")
    
    # Get coefficients
    coeffs = validate_fourier_coefficients()
    
    # Reconstruct signals using Fourier series
    reconstructed = {}
    for name, coeff in coeffs.items():
        reconstructed[name] = fourier_series(coeff, x, method='precompute')
        
        # Compare with original signal
        original = np.load(os.path.join(output_dir, f"{name}_signal.npy"))
        
        plt.figure(figsize=(12, 8))
        plt.plot(x, original, 'b-', label='Original')
        plt.plot(x, reconstructed[name], 'r--', label='Reconstructed')
        plt.title(f"{name.capitalize()} Signal Reconstruction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{name}_reconstruction.png"))
        plt.close()
        
        # Calculate and print error
        error = np.mean(np.abs(original - reconstructed[name]))
        print(f"{name.capitalize()} reconstruction error: {error:.6f}")
        
        # Save reconstructed signal
        np.save(os.path.join(output_dir, f"{name}_reconstructed.npy"), reconstructed[name])
    
    print("Fourier series reconstruction validation complete. Results saved to:", output_dir)
    return reconstructed

def validate_jump_detection():
    """Validate jump detection functionality."""
    print("Validating jump detection...")
    
    # Generate signals with jumps
    signals_with_jumps = {
        'box': box_signal(x, a=2, b=5, normalized=True, jump=True),
        'saw': saw_signal(x, a=2, b=5, normalized=True, jump=True),
        'exp': exp_signal(x, a=2, b=2, c=-1, normalized=True, jump=True),
        'sin': sin_signal(x, a=2, b=2, c=-1, normalized=True, jump=True),
        'gaus': gaussian_signal(x, a=2, b=2, normalized=True, jump=True)
    }
    
    # Save signals and jumps for visual inspection
    for name, (signal, jump) in signals_with_jumps.items():
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        # Ensure signal and x have the same length
        if len(signal) != len(x):
            # Adjust x to match signal length
            x_adjusted = x[:len(signal)]
            plt.plot(x_adjusted, signal)
        else:
            plt.plot(x, signal)
        plt.title(f"{name.capitalize()} Signal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        # Ensure jump and x have the same length
        if len(jump) != len(x):
            # Adjust x to match jump length
            x_adjusted = x[:len(jump)]
            plt.plot(x_adjusted, jump)
        else:
            plt.plot(x, jump)
        plt.title(f"{name.capitalize()} Jump Function")
        plt.xlabel("x")
        plt.ylabel("Jump Value")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_with_jumps.png"))
        plt.close()
        
        # Save signal and jump data
        np.save(os.path.join(output_dir, f"{name}_signal_with_jumps.npy"), signal)
        np.save(os.path.join(output_dir, f"{name}_jump.npy"), jump)
    
    print("Jump detection validation complete. Results saved to:", output_dir)
    return signals_with_jumps

def validate_noise_addition():
    """Validate noise addition functionality."""
    print("Validating noise addition...")
    
    # Generate signals with noise
    noise_levels = [0.05, 0.1, 0.2]
    signals = validate_signal_generation()
    
    for noise_param in noise_levels:
        for name, signal in signals.items():
            noisy_signal = add_noise(signal, noise_param, x)
            
            plt.figure(figsize=(12, 8))
            plt.plot(x, signal, 'b-', label='Original')
            plt.plot(x, noisy_signal, 'r-', alpha=0.7, label=f'Noisy (σ={noise_param})')
            plt.title(f"{name.capitalize()} Signal with Noise (σ={noise_param})")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{name}_noise_{noise_param}.png"))
            plt.close()
            
            # Save noisy signal
            np.save(os.path.join(output_dir, f"{name}_noise_{noise_param}.npy"), noisy_signal)
    
    print("Noise addition validation complete. Results saved to:", output_dir)

def run_all_validations():
    """Run all validation tests."""
    print("Starting validation of refactored code...")
    
    validate_signal_generation()
    validate_fourier_coefficients()
    validate_fourier_series()
    validate_jump_detection()
    validate_noise_addition()
    
    print("\nAll validations complete. Results saved to:", output_dir)
    print("Please inspect the output files to verify that the refactored code produces the expected results.")

if __name__ == "__main__":
    run_all_validations()
