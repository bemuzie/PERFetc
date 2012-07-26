__author__ = 'denis'
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from math import ceil
import ndbilateral


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

def gauss_kernel_3d(sigma,pxsize=None,flsize=None,sigmarule=3):

    if flsize==None:
        pxsize=np.asarray(pxsize)
        fsize=sigmarule*sigma//pxsize
        grid_slices= [slice( -fsize[i]*pxsize[i], fsize[i]*pxsize[i]+pxsize[i], pxsize[i] ) for i in range(3)]
    else:
        flsize=np.asarray(flsize)
        grid_slices= [slice(-i,1+i) for i in flsize//2]
    kernel_grid=np.ogrid[grid_slices]
    kernel_euclid= np.sum( np.power(kernel_grid,2) )
    gauss_kernel=np.exp(kernel_euclid/-(2*sigma**2)) / (np.sqrt(np.pi*2)*sigma)**3
    return gauss_kernel/np.sum(gauss_kernel)

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
def bilateralFunc(data,sigISqrDouble,GausKern,centralpx,kernel_len):
    """ kernel should be  """
    diff=data[centralpx]-data
    coef=np.exp(-diff*diff/sigISqrDouble)*GausKern
    return np.sum(data*coef)/np.sum(coef)


def bilateral3d(img,voxel_size,sigg,sigi,filter=ndbilateral.bilateralFunc):
    """ 4d Bilateral exponential filter.
    image array, kernel size, distance SD, intensity SD
    """

    #optimisation: calculating 2*(Intensity sigma)^2 out from loop
    sigISqrDouble=float(sigi*sigi*2)

    gkern=gauss_kernel_3d(sigg, voxel_size)
    ksize=np.shape(gkern)
    GausKern=np.ravel(gkern)
    kwargs=dict(sigISqrDouble= sigISqrDouble, GausKern= GausKern, centralpx= len(GausKern)/2,kernel_len=len(GausKern))

    img_filtered=ndimage.generic_filter(img,filter,size=ksize+(1,),extra_keywords=kwargs)
    return img_filtered

def bilateral(img,voxel_size,sigg,sigi,mpr=None):
    """
    3d Bilateral exponential filter for 4d volume

    img - image array
    voxel_size - array with x,y,z of voxel
    sigg - distance SD
    sigi - intensity SD
    """
    if not len(voxel_size)==3 and not len(voxel_size)==1:
        raise ValueError("voxel size should be a list of 3 or 1")
    if len(voxel_size)==1:
        voxel_size=np.ones(3)*voxel_size

    if img.ndim==3:
        img=img[...,np.newaxis]

    gaus_kern3d=gauss_kernel_3d(sigg, voxel_size)
    ks_x,ks_y,ks_z=np.asarray(np.shape(gaus_kern3d))/2

    #print np.shape(img),ks_x,ks_y,ks_z

    gaus_kern=np.ravel(gaus_kern3d)
    center=len(gaus_kern)/2
    # calculate 2*sigma^2 of intensity closeness function out from loop
    sigISqrDouble=float(2*sigi*sigi)

    if mpr == None:
        kwargs=dict(sigISqrDouble=sigISqrDouble,GausKern=gaus_kern,center=center)
        ksize=np.shape(gaus_kern3d)
        return ndimage.generic_filter(img,bilateralFunc,size=ksize+(1,),extra_keywords=kwargs)

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

def weightedaverage(data,fsize,sigmadif,dSqrSigma,gaussian):
    diff=np.std(data)-sigmadif
    gc=np.exp(-diff*diff/dSqrSigma)
    coeff=np.ravel(gauss_kernel_3d(gaussian*gc,flsize=fsize))
    return np.sum(data*coeff)/np.sum(coeff)