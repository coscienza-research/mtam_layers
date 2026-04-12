from pathlib import Path

from setuptools import setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="mtam-layers",
    version="0.1.0",
    description="Core differentiable layers for Multi-modal Transformer Associative Memory (mTAM).",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Andrea Ladiana",
    license="CC-BY-4.0",
    python_requires=">=3.10",
    packages=["mtam_layers", "mtam_layers.training"],
    package_dir={"mtam_layers": ".", "mtam_layers.training": "training"},
    package_data={"mtam_layers": ["py.typed"]},
    include_package_data=True,
    install_requires=["torch>=2.0"],
    extras_require={"dev": ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.11.0"]},
)
