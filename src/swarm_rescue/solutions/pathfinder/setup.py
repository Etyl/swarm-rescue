from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("src/pathfinder_c.pyx")
)

# $ python setup.py build_ext --inplace