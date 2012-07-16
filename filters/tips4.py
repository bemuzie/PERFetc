# -*- coding: utf-8 -*-
"""
#TIPS filter described in "TIPS bilateral noise reduction
in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
"""
import numpy as np
def gauss_kernel(size,sigma):
    """Gaussian symetric 4d clousnes kernel """
    kern=np.ones((size,size,size))
    iterArray=kern.copy()

    for i,val in np.ndenumerate(iterArray):
        differ=np.array(i)-np.array([size/2,size/2,size/2])
        euclid=np.sqrt(np.sum(differ*differ))
        kern[i]= np.exp(-0.5*(euclid/sigma)*(euclid/sigma))
    kern=kern[...,None]
    return kern
def TimeProfile_cl(x,lenx,y,sigma):
    """Time profile clousness function. x and y should have shape= (1,1,1,len(time))"""
    diff=x-y
    SSD=np.sum(diff*diff,axis=-1)/lenx
    kern=np.exp(-0.5*(SSD/sigma)*(SSD/sigma))/lenx
    kern=kern[...,None]
    return kern

def convolve4d(img,size,sigG,sigT):
    """Convolve 4d array with symmetric 4d kernel with size "dim_c" through all temporal axis.
    kernel should be odd
    """
    est=np.prod(np.asarray(np.shape(img[...,0]))-size+1)
    made=0
    print 'во время фильтрации будет осуществлено', est, 'циклов'
    sigG=float(sigG)
    sigT=float(sigT)

    #if kernel size even break it
    if size%2 == 0:
        raise NameError('kernel should have odd size!!!')
    #
    size_half=int(size)/2
    #Calculating Gaussian closeness function
    GausKern=gauss_kernel(size,sigG)
    print GausKern
    out=np.zeros((np.shape(img)))
    #making iterator which don't contain borders
    it = np.nditer (img[size_half:-size_half,size_half:-size_half,size_half:-size_half,0], flags=['c_index','multi_index'   ])
    summing = np.sum
    lenX=np.shape(img[0,0,0])
    while not it.finished:

        #determing cordinates of central pixel
        x,y,z=it.multi_index
        center=(x+size_half,y+size_half,z+size_half)
        #Taking kernel of a volume
        kernel = img[x:x+size,y:y+size,z:z+size]
        #Calculating Time profile closeness function
        tp=TimeProfile_cl(img[center],lenX,kernel,sigT)
        #print kernel.shape,GausKern.shape,tp.shape
        coef=tp*GausKern
        out[center]=summing(summing(summing(  coef*kernel,axis=0  ),axis=0),axis=0)/summing(coef)

        it.iternext()
    return out

