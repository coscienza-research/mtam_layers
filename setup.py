from pathlib import Path

from setuptools import setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
VERSION_NS: dict[str, str] = {}
exec(Path(__file__).with_name("version.py").read_text(encoding="utf-8"), VERSION_NS)

setup(
    name="mtam-layers",
    version=VERSION_NS["__version__"],
    description="Core differentiable layers for Multi-modal Transformer Associative Memory (mTAM).",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/andrea-ladiana/mtam_layers",
    project_urls={
        "Repository": "https://github.com/andrea-ladiana/mtam_layers",
        "Issues": "https://github.com/andrea-ladiana/mtam_layers/issues",
        "Documentation": "https://github.com/andrea-ladiana/mtam_layers/blob/master/README.md",
    },
    author="Andrea Ladiana",
    author_email="andrea.ladiana@gmail.com",
    license="Apache-2.0",
    keywords=[
        "associative memory",
        "modern hopfield",
        "multimodal",
        "attention",
        "pytorch",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["mtam_layers", "mtam_layers.training"],
    package_dir={"mtam_layers": ".", "mtam_layers.training": "training"},
    package_data={"mtam_layers": ["py.typed"]},
    include_package_data=True,
    install_requires=["torch>=2.0"],
    extras_require={
        "dev": ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.11.0", "mypy>=1.10.0"],
    },
)
