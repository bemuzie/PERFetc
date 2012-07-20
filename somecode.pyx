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
cdef gausskernel(double sigma,double[3] pxsize):

    fsize=3*sigma//pxsize
    grid_slices=[slice(-fsize[i]*pxsize[i],fsize[i]*pxsize[i],pxsize[i]) for i in range(3)]
    kernel_grid=np.ogrid[grid_slices[0],grid_slices[1],grid_slices[2]]
    kernel_grid*=kernel_grid
    kernel_euclid=np.sqrt( np.sum( np.power(kernel_grid,) ) )
    kernel_gauss=kernel_euclid#some calculations
    return


def prgauss(sigma,np.ndarray [DTYPEfloat_t] pxsize):
    cdef double *pxsizep=<double *>pxsize.data
    gausskernel(sigma,pxsizep)
def printarray(np.ndarray [DTYPEfloat_t, ndim=2] data):
    cdef int length = data.size
    for i in range(1,length-1):
        changearray(data[i-1:i+1,i-1:i+1])
        print i,'loop',data[i-1:i+1,i-1:i+1]


