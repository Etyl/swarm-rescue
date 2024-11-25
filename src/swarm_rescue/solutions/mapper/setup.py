from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

def build():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    setup(
        ext_modules = cythonize(
            "./src/grid_c.pyx",
            force=True,
            compiler_directives={'language_level' : "3"}
        ),
        include_dirs=[numpy.get_include()],
        script_args=['build_ext', '--inplace', '--build-lib', './']
    )

if __name__ == '__main__':
    build()
