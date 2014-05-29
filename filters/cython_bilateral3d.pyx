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
def bilateral3d(int [:,:,:] data, voxel_size, float sigg, float sigi, x_range=[0,-1], y_range=[0,-1], z_range=[0,-1]):
    if data.ndim<3:
        raise ValueError("Input image should have 4 dimensions")


    cdef int imgSize_x=data.shape[0]
    cdef int imgSize_y=data.shape[1]
    cdef int imgSize_z=data.shape[2]
    cdef int from_x = x_range[0], to_x = x_range[1], from_y = y_range[0], to_y = y_range[1], from_z= z_range[0], to_z= z_range[1]



    cdef double value

       
    cdef float [:,:,:] gaus_kern3d = gauss_kernel_3d(sigg, voxel_size)
    
    #cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] gaus_kern3d=np.asarray(gauss_kernel_3d(sigg, voxel_size),dtype=DTYPEfloat)
    cdef int kernelSize_x=gaus_kern3d.shape[0], kside_x=kernelSize_x // 2
    cdef int kernelSize_y=gaus_kern3d.shape[1], kside_y=kernelSize_y // 2
    cdef int kernelSize_z=gaus_kern3d.shape[2], kside_z=kernelSize_z // 2
    """
    if from_x-kside_x < 0 or from_y-kside_y < 0 \
                          or from_z-kside_z < 0 \
                          or to_x+kside_x>imgSize_x \
                          or to_y+kside_y>imgSize_y \
                          or to_z+kside_z>imgSize_z:
        raise ValueError("Kernel ouside image bounds! You should specifi")
    """
    cdef int[:,:,:] data2 = data[from_x-kside_x:to_x+kside_x, from_y-kside_y:to_y+kside_y, from_z-kside_z:to_z+kside_z]


    
    cdef float [:,:,:] result=np.zeros([to_x-from_x,to_y-from_y,to_z-from_z],dtype=DTYPEfloat)
    

    # calculate 2*sigma^2 of intensity closeness function out from loop
    cdef double sigi_double_sqr=2*sigi**2
    cdef double weight_i, weights
    cdef int central_voxel, kernel_voxel
    cdef double gauss_voxel
    cdef unsigned int x,y,z,
    cdef int xk,yk,zk
    cdef double result_value
    cdef DTYPEfloat_t low_border = -100
    cdef DTYPEfloat_t up_border = 200

    cdef float[::1] exp_values= np.asarray( np.exp(-(np.arange(4000)**2)/sigi_double_sqr),
                                            dtype = DTYPEfloat,
                                            order = 'C'
                                            )

    cdef int intensity_difference

    print kside_x, kside_y, kside_z
    #print exp_values.shape[0]
    #print from_x, to_x, from_y, to_y, from_z, to_z
    #print [data2.shape[i] for i in range(3)]
    #print [result.shape[i] for i in range(3)]
    #print range(kside_z, kside_z+to_z-from_z)
    for z in range(kside_z, kside_z+to_z-from_z):
        for x in range(kside_x, kside_x+to_x-from_x):
        
            for y in range(kside_y, kside_y + to_y-from_y):
                #print x,y,z
                
                
                value = 0.0
                weights=0.0
                central_voxel=data2[x,y,z]

                
                for yk in range(-kside_y,kside_y+1):
                    for xk in range(-kside_x,kside_x+1):
                        for zk in range(-kside_z,kside_z+1):

                            kernel_voxel = data2[<unsigned int>(x+xk),<unsigned int>(y+yk),<unsigned int>(z+zk)]
                            gauss_voxel = gaus_kern3d[<unsigned int>(xk+kside_x), <unsigned int> (yk+kside_y), <unsigned int> (zk+kside_z)]
                            intensity_difference = abs(kernel_voxel-central_voxel)
                            #print intensity_difference
                            weight_i = gauss_voxel * exp_values[intensity_difference]
                            
                            value+=kernel_voxel*weight_i
                            #if x%10==0 and y%10==0 and z%100 == 0 :
                            #   print weight_i, data[x+xk,y+yk,z+zk]
                            weights+=weight_i
                            #print z,x,y,'diff',kernel_voxel,central_voxel,intensity_difference,value,gauss_voxel, exp_values[intensity_difference]
                result[x-kside_x,y-kside_y,z-kside_z]= value/weights

    return result
