from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("src/swarm_rescue/solutions/mapper/grid_c.pyx"),
    include_dirs=[numpy.get_include()]
)

# $ python setup.py build_ext --inplace --build-lib /path/to/output_directory
