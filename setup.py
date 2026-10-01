import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "aligner_core",
        ["src/aligner_core.cpp"],
        cxx_std=17,
        extra_compile_args=["/O2"] if sys.platform.startswith("win") else ["-O3"],
    ),
]

setup(
    name="genomic-sequence-aligner",
    version="0.2.0",
    description="Pairwise sequence alignment (NW, SW, Gotoh, Hirschberg) with a pybind11 C++ core",
    python_requires=">=3.9",
    py_modules=["aligner", "align_cli"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    entry_points={"console_scripts": ["seqalign=align_cli:main"]},
)
