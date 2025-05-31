"""
Example script demonstrating model training and evaluation.

This example shows how to generate a dataset, train a model on Fourier coefficients,
and evaluate its performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from fourier_classification.utils import create_domain, prepare_dataset
from fourier_classification.models import (
    create_feed_forward_model, 
    train_model, 
    evaluate_model,
    prepare_data_for_model_b
)
from fourier_classification.visualization import plot_confusion_matrix, plot_training_history

# Create domain
domain = create_domain(start=-np.pi, end=np.pi, num_points=1500)

# Prepare dataset
signal_types = ['Box', 'Saw', 'Exp', 'Sin', 'Gaus']
print("Generating signals dataset...")
signals, labels = prepare_dataset(
    signal_types, 
    num_per_type=100, 
    domain=domain, 
    fourier=True, 
    n_modes=40
)
print(f"Generated {len(signals)} signals with {len(signal_types)} classes")

# Prepare data for Model B (Fourier coefficients)
x_train, x_test, y_train, y_test = prepare_data_for_model_b(signals, labels)
print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

# Create and train model
print("Creating and training model...")
model = create_feed_forward_model(input_shape=(x_train.shape[1],))
model.summary()

model, history = train_model(
    model, 
    x_train, 
    y_train, 
    epochs=10,  # Reduced for example
    batch_size=32, 
    verbose=1, 
    target_accuracy=0.95,  # Reduced for example
    max_iterations=5  # Reduced for example
)

# Plot training history
fig = plot_training_history(history)
plt.savefig("training_history.png")
plt.close(fig)

# Evaluate model
print("Evaluating model...")
results = evaluate_model(model, x_test, y_test, class_names=signal_types)
print(f"Test accuracy: {results['accuracy']:.4f}")

# Plot confusion matrix
fig = plot_confusion_matrix(
    results['confusion_matrix'], 
    class_names=signal_types, 
    title="Confusion Matrix"
)
plt.savefig("confusion_matrix.png")
plt.close(fig)

# Print per-class accuracy
print("\nPer-class accuracy:")
for class_name, accuracy in results['per_class_accuracy'].items():
    print(f"{class_name}: {accuracy:.4f}")

print("\nModel evaluation complete.")
