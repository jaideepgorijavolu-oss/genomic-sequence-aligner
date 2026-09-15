from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "aligner_core",
        ["src/aligner_core.cpp"],
        extra_compile_args=["-O3"] if not __import__("sys").platform.startswith("win") else ["/O2"],
    ),
]

setup(
    name="aligner_core",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)