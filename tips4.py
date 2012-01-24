# -*- coding: utf-8 -*-
"""
#TIPS filter described in "TIPS bilateral noise reduction
in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
"""
import numpy as np
from scipy import ndimage
def gauss_cl(x,sigma):
    """Gaussian clousness function. x and y should be Cartessian coordinates"""
    differ=np.array(x)-np.array([1,1,1])
    euclid=np.sqrt(np.sum(differ*differ))
    return np.exp(-0.5*(euclid/sigma)*(euclid/sigma))
def TimeProfile_cl(x,y,sigma):
    """Time profile clousness function. x and y should have shape= (1,1,1,len(time))"""
    diff=y-x
    SSD=np.sum(diff*diff)/len(x)
    return np.exp(-0.5*(SSD/sigma)*(SSD/sigma))


def convolve4d(img,dim_c,sigG,sigT):
    """Convolve 4d array with 4d kernel with symmetric Cartesian size "dim_c" thought all temporal axis
    """
    #iterating to get Cartesian coordinates each pixel in 'img' of kernel center
    it = np.nditer (img[:,:,:,0], flags=['multi_index'])
    GausKern=np.ones((dim_c,dim_c,dim_c))
    iterArray=np.ones((dim_c,dim_c,dim_c))
    for i,val in np.ndenumerate(iterArray):
        GausKern[i]=gauss_cl(i,sigG)
    GausKern=GausKern[:,:,:,np.newaxis]
    out=np.zeros((np.shape(img)))
    while not it.finished:
        cntr=it.multi_index
        x=cntr[0]
        y=cntr[1]
        z=cntr[2]
        #Taking kernel of a volume
        kernel = img[x-dim_c:x+dim_c,y-dim_c:y+dim_c,z-dim_c:z+dim_c]
        #Calculating Time profile closeness function
        tp=TimeProfile_cl(img[it.multi_index],kernel,sigT)
        #Calculating Gaussian closeness function

        out[cntr]=np.sum(np.sum(np.sum(  img[cntr]*tp*GausKern,axis=0  ),axis=1),axis=2)

        print it.multi_index
        it.iternext()
