# -*- coding: utf-8-*-
# cython: profile=False
 #   Copyright 2012 Denis Nesterov
#   cireto@gmail.com
#
#   The software is licenced under BSD licence.
"""
  A cython implementation of bilateral filtering.
"""

import numpy as np
cimport numpy as np
cimport cython

from cython cimport view


from libc.stdlib cimport abs as c_abs

DTYPE= np.int
DTYPEfloat = np.float32
ctypedef np.int_t DTYPE_t
ctypedef np.float32_t DTYPEfloat_t

cdef extern from "math.h":
     double exp(float x)
cdef extern from "math.h":
    int abs(float x)



def gauss_kernel_3d(sigma,voxel_size):
    sigma = float(sigma)
    # make 3d gauss kernel adaptive to voxel size
    voxel_size=np.asarray(voxel_size, dtype=float)
    # calculate kernel size as distance*sigma from centre
    x,y,z=np.ceil(3*sigma/voxel_size)
    #make 3d grid of euclidean distances from center
    distances=voxel_size*np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    #print distances
    distances=np.sqrt(np.sum( distances*distances ))
    ret = np.exp( distances/ (-2*(sigma**2)) )
    #/ np.sqrt( np.pi*2*sigma**2)**3
    return np.asarray(ret, dtype=DTYPEfloat, order='C')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_weight(double img_value1, double img_value2, double gauss_weight, double sigi_double_sqr):
    return gauss_weight*exp( -((img_value1-img_value2)**2)/ sigi_double_sqr )



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
#sme as data_2 but different looping/
def dict_calc(int [:,:,:] data, float [:,:,:] dist_kernel):
    if data.ndim<3:
        raise ValueError("Input image should have 4 dimensions")


    cdef int imgSize_x=data.shape[0]
    cdef int imgSize_y=data.shape[1]
    cdef int imgSize_z=data.shape[2]



    cdef float value




    #cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] gaus_kern3d=np.asarray(gauss_kernel_3d(sigg, voxel_size),dtype=DTYPEfloat)
    cdef int kernelSize_x=dist_kernel.shape[0], kside_x=kernelSize_x // 2
    cdef int kernelSize_y=dist_kernel.shape[1], kside_y=kernelSize_y // 2
    cdef int kernelSize_z=dist_kernel.shape[2], kside_z=kernelSize_z // 2


    cdef int from_x, to_x, from_y, to_y, from_z, to_z

    from_x, from_y, from_z, to_x,to_y,to_z = 0,0,0,imgSize_x,imgSize_y,imgSize_z






    """
    if from_x-kside_x < 0 or from_y-kside_y < 0 \
                          or from_z-kside_z < 0 \
                          or to_x+kside_x>imgSize_x \
                          or to_y+kside_y>imgSize_y \
                          or to_z+kside_z>imgSize_z:
        raise ValueError("Kernel ouside image bounds! You should specifi")
    """

    cdef float [:,:,:] result=np.zeros([imgSize_x,imgSize_y,imgSize_z],dtype=DTYPEfloat)


    # calculate 2*sigma^2 of intensity closeness function out from loop
    cdef int kernel_voxel
    cdef float dist_voxel, distance
    cdef float max_dist = np.max(dist_kernel)

    cdef unsigned int x,y,z,z_zk,x_xk,y_yk
    cdef int xk,yk,zk



    print kside_x, kside_y, kside_z
    #print exp_values.shape[0]
    #print from_x, to_x, from_y, to_y, from_z, to_z
    #print [data2.shape[i] for i in range(3)]
    #print [result.shape[i] for i in range(3)]
    #print range(kside_z, kside_z+to_z-from_z)
    for z in range(0, imgSize_z-1):
        for x in range(0, imgSize_x-1):
            for y in range(0, imgSize_y-1):




                if data[<unsigned int>(x),<unsigned int>(y),<unsigned int>(z)] != 0:
                    distance = max_dist

                    for yk in range(-kside_y,kside_y+1):
                        for xk in range(-kside_x,kside_x+1):
                            for zk in range(-kside_z,kside_z+1):
                                x_xk=x+xk
                                y_yk=y+yk
                                z_zk=z+zk
                                if 0<=x_xk<=imgSize_x and 0<=y_yk<=imgSize_y and 0<=z_zk<=imgSize_z:

                                    kernel_voxel = data[<unsigned int>x_xk,<unsigned int>y_yk,<unsigned int>z_zk]

                                    dist_voxel = dist_kernel[<unsigned int>(xk+kside_x), <unsigned int> (yk+kside_y), <unsigned int> (zk+kside_z)]

                                    if kernel_voxel==0:
                                        if dist_voxel<distance:
                                            distance = dist_voxel
                                    #print data2[<unsigned int>(x),<unsigned int>(y),<unsigned int>(z)]
                                    #print distance
                    result[x,y,z]= distance
    print 1
    return result

