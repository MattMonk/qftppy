# qftppy

A native PyTorch implementation of the [C++ qft++ library](https://github.com/jdalseno/qft), providing the spinor and tensor objects needed to build covariant tensor amplitudes.

```python
import qftppy

qft = qftppy.QFTNative()
gamma = qftppy.dirac_gamma()
```

## Installation

```bash
pip install qftppy
```

To install from a checkout instead, run `pip install .` in this directory (or `pip install -e .` for an editable install if you expect to make changes to the package).

## Developing

Development is managed with [pixi](https://pixi.sh), which sets up Python, PyTorch and the dev tools from conda-forge and installs the package in editable mode. Common tasks are

```bash
pixi run test   # run the test suite with pytest
pixi run lint   # check linting and formatting with ruff
pixi run fmt    # apply ruff fixes and formatting
pixi run build  # build the sdist and wheel into dist/
```

`pixi shell -e dev` gives a shell in the development environment.

### Comparing against the original C++ qft++

The `cpp` environment builds the original [qft++](https://github.com/jdalseno/qft) from source (pinned commit) together with small pybind11 wrappers in `cpp_reference/`, and runs `tests/cpp/`, which checks qftppy against it numerically:

```bash
pixi run -e cpp test-cpp
```

After changing anything in `cpp_reference/`, rebuild it with `pixi reinstall -e cpp cpp_reference`.

The build applies one local patch to qft++ (`cpp_reference/patches/tensorindex-permute.patch`): upstream `TensorIndex::Permute()` never terminates for rank 4, which hangs `Tensor::Symmetric()` and therefore orbital tensors of rank 4 and above. Regenerate it with `cpp_reference/patches/make_patch.py` rather than editing it by hand.

Both test sets run in GitHub Actions on pushes to main and on pull requests.

To keep the code-base clean, enable `pre-commit` with

```bash
pixi run -e dev pre-commit install
```

This runs several checks plus `ruff` linting and formatting on each commit. The max line length is 120 characters.

## Releasing

Releases are automated with [release-please](https://github.com/googleapis/release-please) and published to [PyPI](https://pypi.org/project/qftppy/) with trusted publishing:

1. Merge changes into `main` as squash-merged pull requests whose titles follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix: ...` for a patch release, `feat: ...` for a minor release, `feat!: ...` for a breaking change, which bumps the minor version while below 1.0; `chore:`, `docs:`, `test:`, `ci:` etc. do not trigger a release). A CI check enforces the title format.
2. release-please keeps a release pull request open that bumps the version in `pyproject.toml` and updates `CHANGELOG.md`.
3. Merging the release pull request tags the release, creates a GitHub release and publishes the package to PyPI.
