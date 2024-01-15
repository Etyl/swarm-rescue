from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("src/swarm_rescue/solutions/mapper/src/grid_c.pyx", force=True),
    include_dirs=[numpy.get_include()],
    script_args=['build_ext', '--inplace', '--build-lib', 'src/swarm_rescue/solutions/mapper/src/']
)

# $ python setup.py build_ext --inplace --build-lib /path/to/output_directory
