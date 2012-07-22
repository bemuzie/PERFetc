import numpy as np
cimport numpy as np
DTYPEfloat = np.float64
ctypedef np.float64_t DTYPEfloat_t

def sumnum(int a, int b,int times):
    cdef int result=0
    cdef unsigned int i
    for i from 0 <= i < times:
        result+=a+b
    return result
