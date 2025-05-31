# How to Upload Your Package to PyPI

This document guides you through uploading your Python package to the Python Package Index (PyPI).

## Prerequisites

1.  **A PyPI Account:** If you don't have one, create an account on [pypi.org](https://pypi.org/).
2.  **Build Your Package:** Ensure you have built the distribution files (sdist and wheel). You can use the provided script:
    ```bash
    ./build_package.sh
    ```
    This will generate your package files in a `dist/` directory. The files will look something like `Fourier_Series_Python-1.0-py3-none-any.whl` and `Fourier-Series-Python-1.0.tar.gz`.

## Installation of Twine

Twine is the recommended tool for securely uploading your packages to PyPI. If you haven't installed it yet, open your terminal and run:

```bash
python -m pip install --user --upgrade twine
```

## Uploading Your Package

1.  **Navigate to Your Project Directory:**
    Open your terminal and change to the root directory of your project (the one containing the `dist/` folder).

2.  **Run Twine Upload:**
    Execute the following command:
    ```bash
    python -m twine upload dist/*
    ```
    This command tells Twine to upload all files from the `dist/` directory.

3.  **Enter Your Credentials:**
    Twine will prompt you for your PyPI username and password.
    *   **Username:** Your PyPI.org username.
    *   **Password:** **It is strongly recommended to use a PyPI API token.**
        *   You can generate an API token from your PyPI account settings (under "API tokens").
        *   When prompted for the password, paste the entire token (including the `pypi-` prefix).
        *   Using a token is more secure as it can be revoked and doesn't expose your main account password.

## Verify on PyPI

After the upload command completes successfully, visit your project page on PyPI to see your new release. For this package, it would be:

[https://pypi.org/project/Fourier-Series-Python/](https://pypi.org/project/Fourier-Series-Python/)

Ensure that version 1.0 is now listed and the description, README, and other metadata look correct.

## (Optional) Using TestPyPI

Before uploading to the official PyPI, you can test the distribution and upload process using TestPyPI ([test.pypi.org](https://test.pypi.org/)). This requires a separate account on TestPyPI.

1.  **Upload to TestPyPI:**
    ```bash
    python -m twine upload --repository testpypi dist/*
    ```
    You'll be prompted for your TestPyPI username and password (or an API token for TestPyPI).

2.  **Install from TestPyPI (to verify):**
    ```bash
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps Fourier-Series-Python
    ```

This helps catch any issues before making a public release.
