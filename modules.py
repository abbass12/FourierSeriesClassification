"""
Visualization and Utility Module
================================

This module provides plotting utilities and helper functions for
visualizing signals, Fourier coefficients, and classification results.

Features:
    - Interactive Plotly-based signal plotting
    - Matplotlib support for static figures
    - Model training history visualization
    - Confusion matrix plotting
    - Fourier coefficient spectrum visualization
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Optional, Union, Tuple
from numpy.typing import NDArray

# Suppress future warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)

# Plotting imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# TensorFlow import (optional for non-ML usage)
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# =============================================================================
# Plotly Visualization Functions
# =============================================================================

def plot(
    fig: go.Figure,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    name: str,
    mode: str = 'lines',
    color: Optional[str] = None
) -> None:
    """
    Add a trace to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure object to add trace to.
    x : NDArray[np.floating]
        X-axis data points.
    y : NDArray[np.floating]
        Y-axis data points.
    name : str
        Name for the trace (appears in legend).
    mode : str, optional
        Plot mode: 'lines', 'markers', or 'lines+markers' (default: 'lines').
    color : str, optional
        Line/marker color (default: auto).
    """
    trace_kwargs = {
        'x': x,
        'y': y,
        'mode': mode,
        'name': name
    }
    if color:
        trace_kwargs['line'] = {'color': color}

    fig.add_trace(go.Scatter(**trace_kwargs))


def plot_signal(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    title: str = "Signal",
    x_label: str = "x",
    y_label: str = "f(x)"
) -> go.Figure:
    """
    Create a Plotly figure for a single signal.

    Parameters
    ----------
    x : NDArray[np.floating]
        Domain points.
    y : NDArray[np.floating]
        Signal values.
    title : str, optional
        Plot title (default: "Signal").
    x_label : str, optional
        X-axis label (default: "x").
    y_label : str, optional
        Y-axis label (default: "f(x)").

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Signal'))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white'
    )
    return fig


def plot_multiple_signals(
    x: NDArray[np.floating],
    signals: List[NDArray[np.floating]],
    names: List[str],
    title: str = "Signals Comparison"
) -> go.Figure:
    """
    Plot multiple signals on the same figure.

    Parameters
    ----------
    x : NDArray[np.floating]
        Domain points (shared for all signals).
    signals : List[NDArray[np.floating]]
        List of signal arrays to plot.
    names : List[str]
        Names for each signal (appears in legend).
    title : str, optional
        Plot title (default: "Signals Comparison").

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    for signal, name in zip(signals, names):
        fig.add_trace(go.Scatter(x=x, y=signal, mode='lines', name=name))

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='f(x)',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_fourier_coefficients(
    cn: NDArray[np.complexfloating],
    title: str = "Fourier Coefficients"
) -> go.Figure:
    """
    Visualize Fourier coefficients (magnitude and phase).

    Parameters
    ----------
    cn : NDArray[np.complexfloating]
        Complex Fourier coefficients.
    title : str, optional
        Plot title (default: "Fourier Coefficients").

    Returns
    -------
    go.Figure
        Plotly figure with magnitude and phase subplots.
    """
    N = len(cn)
    k = np.arange(-N // 2, N // 2 + (N % 2))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Magnitude |c_n|', 'Phase arg(c_n)')
    )

    # Magnitude
    fig.add_trace(
        go.Bar(x=k, y=np.abs(cn), name='Magnitude'),
        row=1, col=1
    )

    # Phase
    fig.add_trace(
        go.Bar(x=k, y=np.angle(cn), name='Phase'),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        template='plotly_white',
        height=600
    )
    fig.update_xaxes(title_text="Mode n", row=2, col=1)

    return fig


def plot_signal_and_reconstruction(
    x: NDArray[np.floating],
    original: NDArray[np.floating],
    reconstructed: NDArray[np.floating],
    title: str = "Signal vs Fourier Reconstruction"
) -> go.Figure:
    """
    Compare original signal with its Fourier reconstruction.

    Parameters
    ----------
    x : NDArray[np.floating]
        Domain points.
    original : NDArray[np.floating]
        Original signal values.
    reconstructed : NDArray[np.floating]
        Reconstructed signal from Fourier coefficients.
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure comparing the signals.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=original,
        mode='lines',
        name='Original',
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=reconstructed,
        mode='lines',
        name='Reconstructed',
        line=dict(dash='dash', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='f(x)',
        template='plotly_white'
    )

    return fig


def plot_accuracy_vs_coefficients(
    n_coefficients: List[int],
    accuracies: List[float],
    title: str = "Classification Accuracy vs Fourier Coefficients"
) -> go.Figure:
    """
    Plot classification accuracy as a function of Fourier coefficient count.

    Parameters
    ----------
    n_coefficients : List[int]
        Number of Fourier coefficients used.
    accuracies : List[float]
        Corresponding classification accuracies (0-100).
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure showing the accuracy trend.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n_coefficients,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        marker=dict(size=10),
        line=dict(width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Number of Fourier Coefficients (N)',
        yaxis_title='Accuracy (%)',
        template='plotly_white',
        yaxis=dict(range=[0, 100])
    )

    return fig


# =============================================================================
# Training Visualization Functions
# =============================================================================

def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss', 'accuracy'],
    title: str = "Training History"
) -> go.Figure:
    """
    Plot training history from a Keras model.

    Parameters
    ----------
    history : dict
        Training history dictionary (from model.fit()).
    metrics : List[str], optional
        Metrics to plot (default: ['loss', 'accuracy']).
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with training curves.
    """
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[m.capitalize() for m in metrics]
    )

    epochs = list(range(1, len(history.get('loss', [])) + 1))

    for i, metric in enumerate(metrics, 1):
        if metric in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history[metric],
                    mode='lines', name=f'Train {metric}'
                ),
                row=i, col=1
            )

        val_metric = f'val_{metric}'
        if val_metric in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history[val_metric],
                    mode='lines', name=f'Validation {metric}',
                    line=dict(dash='dash')
                ),
                row=i, col=1
            )

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=300 * len(metrics)
    )

    return fig


def plot_confusion_matrix(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : NDArray[np.integer]
        True labels.
    y_pred : NDArray[np.integer]
        Predicted labels.
    class_names : List[str], optional
        Names for each class.
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={'size': 14}
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white'
    )

    return fig


# =============================================================================
# Matplotlib Functions (Optional)
# =============================================================================

if HAS_MATPLOTLIB:
    def plot_signal_mpl(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        title: str = "Signal",
        figsize: Tuple[int, int] = (10, 4)
    ) -> plt.Figure:
        """
        Create a matplotlib figure for a signal.

        Parameters
        ----------
        x : NDArray[np.floating]
            Domain points.
        y : NDArray[np.floating]
            Signal values.
        title : str, optional
            Plot title.
        figsize : Tuple[int, int], optional
            Figure size in inches.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, linewidth=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig


# =============================================================================
# Signal Type Names
# =============================================================================

SIGNAL_NAMES = ['Box', 'Saw', 'Exp', 'Sin', 'Gaus']
"""Standard names for the five signal types."""

SIGNAL_COLORS = {
    'Box': '#1f77b4',
    'Saw': '#ff7f0e',
    'Exp': '#2ca02c',
    'Sin': '#d62728',
    'Gaus': '#9467bd'
}
"""Color scheme for consistent signal visualization."""
