__author__ = 'denis'
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from math import ceil


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
def gauss_kernel_3d(sigma,voxel_size,distance=3):
    # make 3d gauss kernel adaptive to voxel size
    voxel_size=np.asarray(voxel_size, dtype=float)
    # calculate kernel size as distance*sigma from centre
    x,y,z=np.ceil(distance*sigma/voxel_size)
    #make 3d grid of euclidean distances from center
    distances=voxel_size*np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    distances*=distances
    distances=np.sqrt(distances[0]+distances[1]+distances[2])

    return np.exp( distances**2/ -2*sigma**2 )/ np.sqrt( np.pi*2*sigma**2 )

def tips(img,voxel_size,sigg,sigt):

    """
    TIPS filter described in "TIPS bilateral noise reduction
    in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
    Convolve 4d array with 3d symmetric kernel through temporal axis.

    voxel_size should be list of 3
    img should be 4d
    """
    if not img.ndim==4:
        raise NameError('image should have 4 dimensions!')
    sigg=float(sigg)
    #Calculating Gaussian kernel
    GausKern=gauss_kernel_3d(sigg,voxel_size)
    #Calculating x,y,z sizes of kernel
    x_ks,y_ks,z_ks=np.shape(GausKern)
    center=(np.shape(GausKern)/2)
    lenT=np.shape(img)[-1]
    img_filtered=np.zeros(np.shape(img))
    #making iterator which don't contain borders
    img_iter=np.ndindex(np.shape(img[x_ks:,y_ks:,z_ks:,0]))
    #Optimizations
    # Calculating 2*sigT**2 out from loop to increase optimize Time closness calculations
    # and making np.sum local
    sigTSqrDouble=float(sigt*sigt*2)
    summ=np.sum
    for x,y,z in img_iter:
        kernel=img[x:x+x_ks, y:y+y_ks, z:z+z_ks]
        diff=kernel[center]-kernel
        SSD=summ(diff*diff,axis=-1)/lenT
        TclsKern=np.exp(-SSD*SSD/sigTSqrDouble)/lenT
        coeff=TclsKern*GausKern
        coeff=coeff[...,None]
        img_filtered[x+center[0],y+center[1],z+center[2]]=summ(summ(summ(kernel*coeff,0),0),0)/summ(coeff)
    return img_filtered
def bilateralFunc(data,sigISqrDouble,GausKern,center):
    """ kernel should be  """
    diff=data[center]-data
    IclsKern=np.exp(-diff*diff/sigISqrDouble)
    coef=IclsKern*GausKern
    a = np.sum(data*coef)/np.sum(coef)
    return a

def bilateral3d(img,voxel_size,sigg,sigi):
    """ 4d Bilateral exponential filter.
    image array, kernel size, distance SD, intensity SD
    """

    #optimisation: calculating 2*(Intensity sigma)^2 out from loop
    sigISqrDouble=float(sigi*sigi*2)

    gkern=gauss_kernel_3d(sigg, voxel_size)
    ksize=np.shape(gkern)
    GausKern=np.ravel(gkern)
    #Closness function
    kwargs=dict(sigISqrDouble=sigISqrDouble,GausKern=GausKern,center=len(GausKern)/2)
    img_filtered=ndimage.generic_filter(img,ndbilateral.bilateralFunc,size=ksize+(1,),extra_keywords=kwargs)
    return img_filtered

def bilateral(img,voxel_size,sigg,sigi,mpr=None):
    """ 3d Bilateral exponential filter for 4d volume
    img - image array, voxel_size - array with x,y,z of voxel; sigg - distance SD; sigi - intensity SD
    """
    dimensions=np.ndim(img)
    if dimensions==3:
        img=img[...,np.newaxis]
    gaus_kern3d=gauss_kernel_3d(sigg, voxel_size)
    ks_x,ks_y,ks_z=np.asarray(np.shape(gaus_kern3d))/2

    #print np.shape(img),ks_x,ks_y,ks_z

    gaus_kern=np.ravel(gaus_kern3d)
    center=len(gaus_kern)/2
    # calculate 2*sigma^2 of intensity closeness function out from loop
    sigISqrDouble=float(2*sigi*sigi)
    #Closness function
    kwargs=dict(sigISqrDouble=sigISqrDouble,GausKern=gaus_kern,center=center)
    if mpr == None:
        ksize=np.shape(gaus_kern3d)
        return ndimage.generic_filter(img,bilateralFunc,size=ksize+(1,),extra_keywords=kwargs)
    print 'bla bla'
    #filtration of selected vol
    slice_iter=np.nditer(img[[slice(i[0],i[1]) for i in mpr]],flags=['c_index','multi_index'])
    outputvol=np.zeros(np.shape(img[[slice(i[0],i[1]) for i in mpr]]))
    zero_coords=[i[0] for i in mpr]
    while not slice_iter.finished:
        if slice_iter.value<-500:
            outputvol[slice_iter.multi_index]=slice_iter.value
            slice_iter.iternext()
            continue
        try:
            x,y,z,t=np.asarray(slice_iter.multi_index)+np.asarray(zero_coords)
            img_kernel=img[x-ks_x:x+1+ks_x,y-ks_y:y+1+ks_y,z-ks_z:z+1+ks_z,t]
            diff=slice_iter.value-img_kernel
            coef=gaus_kern3d*np.exp(-diff*diff/sigISqrDouble)
            outputvol[slice_iter.multi_index]=np.sum(img_kernel*coef)/np.sum(coef)
            slice_iter.iternext()
        except ValueError:
            slice_iter.iternext()
            continue
    return outputvol

