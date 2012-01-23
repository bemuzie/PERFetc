# -*- coding: utf-8 -*-
"""
#TIPS filter described in "TIPS bilateral noise reduction
in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
"""
import numpy as np

def gauss_cl(x,y,sigma):
    """Gaussian clousness function. x and y should be Cartessian coordinates"""
    differ=x-y
    euclid=np.sqrt(differ[0]*differ[0],differ[1]*differ[1],differ[2]*differ[2])
    return np.exp(-0.5*(euclid/sigma)*(euclid/sigma))
def TimeProfile_cl(x,y,sigma):
    """Time profile clousness function. x and y should have shape= (1,1,1,len(time))"""
    SSD=np.sum((x-y)*(x-y))/len(x)
    return np.exp(-0.5*(SSD/sigma)*(SSD/sigma))


def convolve4d(img,dim_c):
    """Convolve 4d array with 4d kernel with symmetric Cartesian size "dim_c" thought all temporal axis
    """
    #iterating to get Cartesian coordinates each pixel in 'img' of kernel center
    it = np.nditer (img[:,:,:,0], flags=['multi_index'])
    area=img[dim_c:dim_c,dim_c:dim_c,dim_c:dim_c]
    kernel=np.ones((dim_c,dim_c,dim_c))
    while not it.finished:
        if it.multindex
        center=img[it.multi_index]
        others=area[it.multi_index+tuple((3,3,3))]
        print it.multi_index
        it.iternext()

a=np.arange(6*6*4*4).reshape(6,6,4,4)
convolve4d(a,3)