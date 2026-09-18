# pyqft_native

A native PyTorch implementation of the [C++ qft++ library](https://github.com/jdalseno/qft), providing the spinor and tensor objects needed to build covariant tensor amplitudes.

## Installation

Within this directory simply

```bash
pip install .
```

You can use `pip install -e .` for an editable install if you expect to make changes to the package.

## Developing

Enable `pre-commit` (`pip install pre-commit`) with

```bash
pre-commit install
```

This runs several checks, a Python formatter (`yapf`) and `flake8` on each commit. Note that the max line length is set to 120 characters.
