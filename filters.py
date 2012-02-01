__author__ = 'denis'
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
def std(image,num):
    sh=np.shape(image)[0]
    for parts in np.arange(num)[::-1]:
        print sh%parts,parts
        if sh%parts == 0:
            partsnum=parts
            break
    it=np.linspace(0,sh,partsnum+1)
    print it
    tips=np.std(image[it[0]:it[1]],axis=-1)
    for ind,val in enumerate(it[2:]):
        tipsplus=np.std(image[it[ind+1]:val],axis=-1)
        tips=np.append(tips,tipsplus,0)
        print it[ind+1],val
    return tips
def gauss_kernel(size,sigma):
    """Gaussian symetric 4d clousnes kernel """
    kern=np.ones((size,size,size))
    iterArray=kern.copy()

    for i,val in np.ndenumerate(iterArray):
        differ=np.array(i)-np.array([size/2,size/2,size/2])
        euclid=np.sqrt(np.sum(differ*differ))
        kern[i]= np.exp(-0.5*(euclid/sigma)*(euclid/sigma))
    return kern


def tips4d(img,size,sigG,sigT):

    """
    TIPS filter described in "TIPS bilateral noise reduction
    in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
    Convolve 4d array with 3d symmetric kernel through all temporal axis.
    kernel should be odd
    """
    #if kernel size even break it
    if size%2 == 0:
        raise NameError('kernel should have odd size!!!')

    est=np.prod(np.asarray(np.shape(img[...,0]))-size+1)
    made=0
    print 'во время фильтрации будет осуществлено', est, 'циклов'
    sigG=float(sigG)
    #Calculating 2*sigT**2 out from loop to increase optimize Time closness calculations
    sigTSqrDouble=float(sigT)*float(sigT)*2

    size_half=int(size)/2
    #Calculating Gaussian kernel
    GausKern=gauss_kernel(size,sigG)
    print GausKern
    out=np.zeros((np.shape(img)))
    #making iterator which don't contain borders
    it = np.nditer (img[size_half:-size_half,size_half:-size_half,size_half:-size_half,0], flags=['c_index','multi_index'   ])
    summing = np.sum
    tAxisLength=np.shape(img[0,0,0])
    while not it.finished:

        #determing cordinates of central pixel
        x,y,z=it.multi_index
        center=(x+size_half,y+size_half,z+size_half)
        kernel=img[x:x+size,y:y+size,z:z+size]
        #Calculating Time profile closeness function.
        diff=img[center]-kernel
        SSD=np.sum(diff*diff,axis=-1)/tAxisLength
        tp=np.exp(-(SSD*SSD)/sigTSqrDouble)/tAxisLength

        #print kernel.shape,GausKern.shape,tp.shape
        coef=tp*GausKern
        coef=coef[...,None]
        out[center]=summing(summing(summing(  coef*kernel,axis=0  ),axis=0),axis=0)/summing(coef)

        it.iternext()
    return out

def TimeProfile_cl(data,sigTSqrDouble,GausKern,center,lenT):
    """Time profile clousness function. x and y should have shape= (1,1,1,len(time))"""

    diff=data[center]-data
    SSD=np.sum(diff*diff,axis=-1)/lenT
    TclsKern=np.exp(-SSD*SSD/sigTSqrDouble)
    print TclsKern[3,3],sigTSqrDouble,'ssd',SSD[3,3]
    coeff=TclsKern*GausKern
    coeff=coeff[...,None]
    data_filtered=np.sum(np.sum(np.sum(data*coeff,0),0),0)/np.sum(coeff)

    return data_filtered
def tips4d_m(img,size,sigG,sigT):

    """
    TIPS filter described in "TIPS bilateral noise reduction
    in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
    Convolve 4d array with 3d symmetric kernel through all temporal axis.
    kernel should be odd
    """
    #if kernel size even break it
    if size%2 == 0:
        raise NameError('kernel should have odd size!!!')

    est=np.prod(np.asarray(np.shape(img[...,0]))-size+1)
    made=0
    print 'во время фильтрации будет осуществлено', est, 'циклов'
    sigG=float(sigG)
    #Calculating 2*sigT**2 out from loop to increase optimize Time closness calculations
    sigTSqrDouble=float(sigT)*float(sigT)*2

    size_half=int(size)/2
    #Calculating Gaussian kernel
    GausKern=gauss_kernel(size,sigG)
    sigTSqrDouble=sigT*sigT*2
    sigTSqrDouble=float(sigTSqrDouble)
    center=(size_half,size_half,size_half)
    lenT=np.shape(img[0,0,0])

    img_filtered=np.zeros((np.shape(img)))
    #making iterator which don't contain borders
    img_shp=np.shape(img[size:,size:,size:,0])

    for x,y,z in np.ndindex(img_shp):
        kernel=img[x:x+size,y:y+size,z:z+size]
        img_filtered[x+size_half,y+size_half,z+size_half]=TimeProfile_cl(kernel,sigTSqrDouble,GausKern,center,lenT)

    return img_filtered
def bilateralFunc(data,sigISqrDouble,GausKern,center=None):
    """ kernel should be  """

    diff=data[center]-data
    IclsKern=np.exp(-diff*diff/sigISqrDouble)
    coef=IclsKern*GausKern
    a = np.sum(data*coef)/np.sum(coef)
    return a

def bilateralFilter(img,size,sigG,sigI):
    """ 4d Bilateral exponential filter.
    image array, kernel size, distance SD, intensity SD
    """
    #img=np.array(img,dtype=float)
    ksize=np.power(size,3)
    center=ksize/2
    sigISqrDouble=sigI*sigI*2
    sigISqrDouble=float(sigISqrDouble)

    GausKern=np.ravel(gauss_kernel(size,sigG))

    #Closness function
    kwargs=dict(sigISqrDouble=sigISqrDouble,GausKern=GausKern,center=center)
    img_filtered=ndimage.generic_filter(img,bilateralFunc,size=[size,size,size,1],extra_keywords=kwargs)
    return img_filtered

def bilateralFilter_t(img,size,sigG,sigI):
    """ 4d Bilateral exponential filter.
    image array, kernel size, distance SD, intensity SD
    """
    #img=np.array(img,dtype=float)


    #if kernel size even break it
    if size%2 == 0:
        raise NameError('kernel should have odd size!!!')

    sigG=float(sigG)
    #Calculating 2*sigT**2 out from loop to increase optimize Time closness calculations
    sigISqrDouble=sigI*sigI*2
    sigISqrDouble=float(sigISqrDouble)
    GausKern=gauss_kernel(size,sigG)

    #Calculating Gaussian kernel
    img_filtered=np.zeros((np.shape(img)))
    size_half=int(size)/2
    #making iterator which don't contain borders

    shp=np.shape(img[size:,size:,size:])

    for i in np.ndindex(shp):
        x,y,z,t =i
        #determing cordinates of central pixel
        kernel=img[x:x+size,y:y+size,z:z+size,t]
        #Calculating Time profile closeness function.

        img_filtered[x+size_half,y+size_half,z+size_half,t]=bilateralFunc(kernel,sigISqrDouble,GausKern,(1,1,1))
        #print kernel.shape,GausKern.shape,tp.shape
        #coef=coef[...,None]

    return img_filtered