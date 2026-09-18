from setuptools import setup, find_packages

setup(
    name="pyqft_native",
    version="0.1.0",
    description="Native PyTorch implementation of QFT++",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["torch>=1.10.0", "numpy"],
    python_requires=">=3.7",
)
