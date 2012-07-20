#   Copyright 2012 Denis Nesterov
#   cireto@gmail.com
#
#   The software is licenced under BSD licence.
"""
  A cython implementation of bilateral filtering.
"""

cimport cython
cimport numpy as np
import numpy as np
DTYPEfloat = np.float64
ctypedef np.float64_t DTYPEfloat_t

cdef extern from "math.h":
    double exp(double x)


def gauss_kernel_3d(sigma,voxel_size):
    #voxel ratio is size in x dimension devided by size in z dimension
    voxel_size=np.asarray(voxel_size,dtype=float)
    # calculate 3 sigma distance from centre
    x,y,z=np.ceil(3*sigma/voxel_size)
    distances=np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    gauss=-0.5*np.multiply(distances,distances)/sigma**2
    gauss3d=1

    for i in gauss: gauss3d = gauss3d*(np.exp(i)/np.sqrt(np.pi*2*sigma**2))
    return gauss3d

@cython.boundscheck(False)
def bilateralFunc(np.ndarray [DTYPEfloat_t, ndim=1] data,DTYPEfloat_t sigISqrDouble,np.ndarray [DTYPEfloat_t, ndim=1] GausKern,int centralpx, int kernel_len):

    cdef DTYPEfloat_t coefsum
    cdef DTYPEfloat_t result
    cdef unsigned int pxnum
    cdef DTYPEfloat_t coef,cpx,px
    cdef DTYPEfloat_t *pdata=<DTYPEfloat_t *>data.data, *pGausKern=<DTYPEfloat_t *>GausKern.data

    coefsum=0
    result=0
    cpx=pdata[centralpx]
    for pxnum in range(kernel_len):
        px=pdata[pxnum]
        coef=exp(-(px-cpx)**2/sigISqrDouble) * pGausKern[pxnum]
        coefsum+=coef
        result+=px * coef
    return result/coefsum

@cython.boundscheck(False)
def bilatfunc_opt(np.ndarray data,double sigISqrDouble,np.ndarray GausKern,int centralpx, int kernel_len):

    cdef double coefsum
    cdef double result
    cdef unsigned int pxnum
    cdef double coef,cpx,px
    cdef double *pdata=<double *>data.data, *pGausKern=<double *>GausKern.data

    coefsum=0
    result=0
    cpx=pdata[centralpx]
    for pxnum in range(kernel_len):
        px=pdata[pxnum]
        coef=exp(-(px-cpx)**2 / sigISqrDouble) * pGausKern[pxnum]
        coefsum+=coef
        result+=px * coef
    return result/coefsum

@cython.cdivision(True)
cdef bilatfunc(np.ndarray [DTYPEfloat_t, ndim=3] datakern,double *gauskern, double dsquaredIntensitySigma,int kernel_size,int cindex):
    cdef double *pdatakern=<double *>datakern.data
    cdef double cpx=pdatakern[cindex]
    cdef unsigned int i
    cdef double coef
    cdef double result=0.0
    cdef double coefsum=0.0

    for i in range(kernel_size):
        px=pdatakern[i]
        coef=exp(-(px-cpx)**2/dsquaredIntensitySigma) * gauskern[i]
        coefsum+=coef
        result+= px * coef
    return result/coefsum

@cython.boundscheck(False)
def bilateral(np.ndarray[DTYPEfloat_t, ndim=4] data,voxel_size,double sigg,double sigi):
    if not data.ndim == 4:
        raise ValueError("Input image should have 4 dimensions")

    assert data.dtype == DTYPEfloat

    cdef int imgSize_x=data.shape[0]
    cdef int imgSize_y=data.shape[1]
    cdef int imgSize_z=data.shape[2]
    cdef int imgSize_t=data.shape[3]

    cdef np.ndarray[DTYPEfloat_t, ndim=3] gaus_kern3d=np.asarray(gauss_kernel_3d(sigg, voxel_size),dtype=DTYPEfloat)
    cdef int kernelSize_x=gaus_kern3d.shape[0]
    cdef int kernelSize_y=gaus_kern3d.shape[1]
    cdef int kernelSize_z=gaus_kern3d.shape[2]
    cdef int kernelSize=gaus_kern3d.size
    cdef double *gauskern = <double *>gaus_kern3d.data
    cdef int kernelCenter=kernelSize//2

    cdef int kside_x=kernelSize_x // 2
    cdef int kside_y=kernelSize_y // 2
    cdef int kside_z=kernelSize_z // 2

    cdef np.ndarray[DTYPEfloat_t, ndim=4] result=np.zeros([imgSize_x,imgSize_y,imgSize_z,imgSize_t],dtype=DTYPEfloat)
    # calculate 2*sigma^2 of intensity closeness function out from loop
    cdef double sigiSqrDouble=2*sigi**2
    cdef unsigned int x,y,z,t


    for x in range(<unsigned int> kside_x, <unsigned int>(imgSize_x - kside_x - 1) ):
        for y in range(<unsigned int> kside_y, <unsigned int>(imgSize_y - kside_y - 1) ):
            for z in range(<unsigned int> kside_z, <unsigned int>(imgSize_z - kside_z - 1) ):
                for t in range(<unsigned int>imgSize_t):
                    result[x,y,z,t]=bilatfunc(data[x:kernelSize_x,y:kernelSize_x,z:kernelSize_x,t],gauskern,sigiSqrDouble,kernelSize,kernelCenter)
    return result

@cython.boundscheck(False)
@cython.cdivision(True)
def bilateral_optimized(np.ndarray[DTYPEfloat_t, ndim=4] data,voxel_size,double sigg,double sigi):
    if not data.ndim == 4:
        raise ValueError("Input image should have 4 dimensions")

    assert data.dtype == DTYPEfloat

    cdef int imgSize_x=data.shape[0]
    cdef int imgSize_y=data.shape[1]
    cdef int imgSize_z=data.shape[2]
    cdef int imgSize_t=data.shape[3]
    cdef double value

    cdef np.ndarray[DTYPEfloat_t, ndim=3] gaus_kern3d=np.asarray(gauss_kernel_3d(sigg, voxel_size),dtype=DTYPEfloat)
    cdef int kernelSize_x=gaus_kern3d.shape[0]
    cdef int kernelSize_y=gaus_kern3d.shape[1]
    cdef int kernelSize_z=gaus_kern3d.shape[2]

    cdef int kside_x=kernelSize_x // 2
    cdef int kside_y=kernelSize_y // 2
    cdef int kside_z=kernelSize_z // 2

    # calculate 2*sigma^2 of intensity closeness function out from loop
    cdef DTYPEfloat_t sigiSqrDouble=2*sigi**2
    cdef double weight_i, weights
    cdef DTYPEfloat_t px
    cdef unsigned int x,y,z,t
    cdef int xk,yk,zk

    cdef np.ndarray[DTYPEfloat_t, ndim=4] result=np.zeros([imgSize_x,imgSize_y,imgSize_z,imgSize_t],dtype=DTYPEfloat)

    for x in range(<unsigned int> kside_x, <unsigned int>(imgSize_x - kside_x - 1) ):
        for y in range(<unsigned int> kside_y, <unsigned int>(imgSize_y - kside_y - 1) ):
            for z in range(<unsigned int> kside_z, <unsigned int>(imgSize_z - kside_z - 1) ):
                for t in range(<unsigned int>imgSize_t):
                    value = 0.0
                    weights=0.0

                    for xk in range(-kside_x, kside_x+1):
                        for yk in range(-kside_y, kside_y+1):
                            for zk in range(-kside_z, kside_z+1):
                                px=data[<unsigned int>(x+xk), <unsigned int>(y+yk), <unsigned int>(z+zk), t]
                                weight_i=gaus_kern3d[<unsigned int> (xk+kside_x), <unsigned int>(yk+kside_y), <unsigned int>(zk+kside_z)]*\
                                         exp( -(px - data[x, y, z, t])**2 / sigiSqrDouble)
                                value+=px * weight_i
                                weights+=weight_i
                    result[x,y,z,t]= value/weights
    return data
