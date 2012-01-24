# -*- coding: utf-8 -*-
"""
#TIPS filter described in "TIPS bilateral noise reduction
in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
"""
import numpy as np
from scipy import ndimage
def gauss_cl(x,sigma):
    """Gaussian clousness function. x and y should be Cartessian coordinates"""
    differ=x-[1,1,1]
    euclid=np.sqrt(differ[0]*differ[0],differ[1]*differ[1],differ[2]*differ[2])
    return np.exp(-0.5*(euclid/sigma)*(euclid/sigma))
def TimeProfile_cl(x,y,sigma):
    """Time profile clousness function. x and y should have shape= (1,1,1,len(time))"""
    diff=x-y
    SSD=np.sum(diff*diff)/len(x)
    return np.exp(-0.5*(SSD/sigma)*(SSD/sigma))


def convolve4d(img,dim_c,sigG):
    """Convolve 4d array with 4d kernel with symmetric Cartesian size "dim_c" thought all temporal axis
    """
    #iterating to get Cartesian coordinates each pixel in 'img' of kernel center
    it = np.nditer (img[:,:,:,0], flags=['multi_index'])
    area=img[dim_c:dim_c,dim_c:dim_c,dim_c:dim_c]
    a=np.ones((dim_c,dim_c,dim_c))
    GausKern=np.ones((dim_c,dim_c,dim_c))
    iterArray=np.ones((dim_c,dim_c,dim_c))
    for i,val in np.ndenumerate(iterArray):
        GausKern[i]=gauss_cl(i,sigG)

    while not it.finished:
        cntr=it.multi_index
        x=cntr[0]
        y=cntr[1]
        z=cntr[2]
        #Taking kernel of a volume
        kernel = img[x-dim_c:x+dim_c,y-dim_c:y+dim_c,z-dim_c:z+dim_c]
        #Calculating Time profile closeness function
        tp=TimeProfile_cl(img[it.multi_index],kernel)
        #Calculating Gaussian closeness function
        img[cntr]

        print it.multi_index
        it.iternext()

a=np.arange(6*6*4*4).reshape(6,6,4,4)
convolve4d(a,3)