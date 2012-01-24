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


def convolve4d(img,size,sigG,sigT):
    """Convolve 4d array with symmetric 4d kernel with size "dim_c" through all temporal axis.
    kernel should be odd
    """

    #if kernel size even break it
    if size%2 == 0:
        raise NameError('kernel should have odd size!!!')
    #
    size_3d=tuple((size,size,size))
    size_half=int(size)/2
    #
    GausKern=np.ones(size_3d)
    iterArray=GausKern.copy()
    #
    for i,val in np.ndenumerate(iterArray):
        GausKern[i]=gauss_cl(i,sigG)
    GausKern=GausKern[:,:,:,np.newaxis]
    #
    out=np.zeros((np.shape(img)))
    #making iterator which don't contain borders
    it = np.nditer (img[size_half:-size_half,size_half:-size_half,size_half:-size_half,0], flags=['multi_index'])
    while not it.finished:
        #determing cordinates of central pixel
        cent=it.multi_index
        x=cent[0]
        y=cent[1]
        z=cent[2]
        center=tuple((x+size_half,y+size_half,z+size_half))
        endboder=size+1
        #Taking kernel of a volume
        kernel = img[x:endboder,y:endboder,z:endboder]
        #Calculating Time profile closeness function
        tp=TimeProfile_cl(img[center],kernel,sigT)
        #Calculating Gaussian closeness function

        out[center]=np.sum(np.sum(np.sum(  kernel*tp*GausKern,axis=0  ),axis=0),axis=0)

        print it.multi_index
        it.iternext()
