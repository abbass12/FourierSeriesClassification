from setuptools import setup, find_packages

setup(
    name="fourier_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "plotly>=4.14.0",
        "tensorflow>=2.4.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
    ],
    author="Abbass Srour",
    author_email="abbasss@umich.edu",
    description="A package for classifying 1D signals using Fourier Series and Machine Learning",
    keywords="fourier, signal processing, machine learning, classification",
    python_requires=">=3.6",
)
