 #   Copyright 2012 Denis Nesterov
#   cireto@gmail.com
#
#   The software is licenced under BSD licence.
"""
  A cython implementation of bilateral filtering.
"""


import numpy as np
cimport numpy as np
DTYPE= np.int
DTYPEfloat = np.float64
ctypedef np.int_t DTYPE_t
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
    print np.shape(gauss3d)
    return gauss3d

"""
def bilateralFunc(np.ndarray data,double sigISqrDouble,np.ndarray GausKern,int center):
    cdef int length = data.shape
    cdef double result, weights, weight_i
    cdef double *pdata=<double *>data.data
    cdef double *kdata=<double *>GausKern.data


    diff=data[center]-data
    IclsKern=np.exp(-diff*diff/sigISqrDouble)
    coef=IclsKern*GausKern

    for i in range(length):
        data_i=data[i]
        weight_i=GausKern[i]*exp((-data[center]-data_i**2)/sigISqrDouble)
        weights+=weight_i
        result+=data_i*weight_i

    return np.sum(data*coef)/np.sum(coef)
"""
def bilateral(np.ndarray[DTYPEfloat_t, ndim=4] data,voxel_size,double sigg,double sigi):
    if data.ndim<3:
        raise ValueError("Input image should have 4 dimensions")

    assert data.dtype == DTYPEfloat

    cdef int imgSize_x=data.shape[0]
    cdef int imgSize_y=data.shape[1]
    cdef int imgSize_z=data.shape[2]
    cdef int imgSize_t=data.shape[3]
    cdef DTYPEfloat_t value


    cdef np.ndarray[DTYPEfloat_t, ndim=3] gaus_kern3d=np.asarray(gauss_kernel_3d(sigg, voxel_size),dtype=DTYPEfloat)
    cdef int kernelSize_x=gaus_kern3d.shape[0]
    cdef int kernelSize_y=gaus_kern3d.shape[1]
    cdef int kernelSize_z=gaus_kern3d.shape[2]

    cdef int kside_x=kernelSize_x // 2
    cdef int kside_y=kernelSize_y // 2
    cdef int kside_z=kernelSize_z // 2

    cdef np.ndarray[DTYPEfloat_t, ndim=4] result=np.zeros([imgSize_x,imgSize_y,imgSize_z,imgSize_t],dtype=DTYPEfloat)
    cdef int resultSize_x = imgSize_x
    # calculate 2*sigma^2 of intensity closeness function out from loop
    cdef double sigiSqrDouble=2*sigi**2
    cdef double weight_i, weights
    cdef DTYPEfloat_t diff
    cdef int x,y,z,t,xk,yk,zk
    for x in range(kside_x, imgSize_x - kside_x - 1):
        for y in range(kside_y, imgSize_y - kside_y - 1):
            for z in range(kside_z, imgSize_z - kside_z - 1):
                for t in range(imgSize_t):
                    value = 0.0
                    weights=0
                    for xk in range(-kside_x,kside_x+1):
                        for yk in range(-kside_y,kside_y+1):
                            for zk in range(-kside_z,kside_z+1):
                                weight_i=gaus_kern3d[xk+kside_x,yk+kside_y,zk+kside_z]*\
                                         exp( -(data[x+xk,y+yk,z+zk,t] - data[x,y,z,t])**2 / sigiSqrDouble)
                                value+=data[x+xk,y+yk,z+zk,t]*weight_i
                                weights+=weight_i
                    result[x,y,z,t]= value/weights
    return result