import numpy as np
cimport numpy as np
DTYPEfloat = np.float64
ctypedef np.float64_t DTYPEfloat_t


cdef changearray(np.ndarray dkern):
    cdef int length = dkern.size
    cdef double *pkern=<double *>dkern.data
    print "new loop"
    for i in range(length):
        print pkern[i]


def printarray(np.ndarray [DTYPEfloat_t, ndim=2] data):
    cdef int length = data.size
    for i in range(1,length-1):
        changearray(data[i-1:i+1,i-1:i+1])
        print i,'loop',data[i-1:i+1,i-1:i+1]


