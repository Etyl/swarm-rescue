import numpy as np
import cython

cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

