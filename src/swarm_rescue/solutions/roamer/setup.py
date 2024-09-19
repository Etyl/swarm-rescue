from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    ext_modules = cythonize("src/frontier_explorer_c.pyx", force=True),
    include_dirs=[numpy.get_include()],
    script_args=['build_ext', '--inplace', '--build-lib', './']
)
