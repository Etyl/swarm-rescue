from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    ext_modules = cythonize(
        "src/pathfinder_c.pyx",
        force=True,
        compiler_directives={'language_level' : "3"}
    ),
    include_dirs=[numpy.get_include()],
    script_args=['build_ext', '--inplace', '--build-lib', './'],
)
