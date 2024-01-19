from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("src/pathfinder_c.pyx", force=True),
    include_dirs=[numpy.get_include()],
    script_args=['build_ext', '--inplace', '--build-lib', './']
)
