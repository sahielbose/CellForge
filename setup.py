from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="cellforge",
    version="0.2.0",
    description="Agentic design of virtual cell models for single-cell perturbation data",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/sahielbose/CellForge",
    license="MIT",
    packages=find_packages(include=["cellforge", "cellforge.*"]),
    py_modules=["main"],
    entry_points={"console_scripts": ["cellforge=main:cli_entrypoint"]},
    install_requires=REQUIREMENTS,
    extras_require={"pdf": ["pymupdf>=1.22"], "dev": ["pytest>=7.0", "ruff>=0.4"]},
    python_requires=">=3.9,<3.13",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
