#!/bin/bash
# Script to build the Fourier-Series-Python package

echo "Ensuring build tools (pip, setuptools, wheel, build) are up-to-date/installed..."
python -m pip install --user --upgrade pip setuptools wheel build

echo "Removing old build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf fourier_classification.egg-info # Also remove specific egg-info if present

echo "Building the package (sdist and wheel)..."
python -m build

echo ""
echo "Package built successfully!"
echo "You can find the distribution files in the 'dist/' directory."
echo "These files are ready for upload to PyPI using twine."
