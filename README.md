# qftppy

A native PyTorch implementation of the [C++ qft++ library](https://github.com/jdalseno/qft), providing the spinor and tensor objects needed to build covariant tensor amplitudes.

```python
import qftppy

qft = qftppy.QFTNative()
gamma = qftppy.dirac_gamma()
```

## Installation

With pip, from within this directory

```bash
pip install .
```

Use `pip install -e .` for an editable install if you expect to make changes to the package.

## Developing

Development is managed with [pixi](https://pixi.sh), which sets up Python, PyTorch and the dev tools from conda-forge and installs the package in editable mode. Common tasks are

```bash
pixi run test   # run the test suite with pytest
pixi run lint   # check linting and formatting with ruff
pixi run fmt    # apply ruff fixes and formatting
pixi run build  # build the sdist and wheel into dist/
```

`pixi shell -e dev` gives a shell in the development environment.

To keep the code-base clean, enable `pre-commit` with

```bash
pixi run -e dev pre-commit install
```

This runs several checks plus `ruff` linting and formatting on each commit. The max line length is 120 characters.
