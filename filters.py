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
def gauss_kernel_3d(sigma,voxel_size):
    voxel_size=np.asarray(voxel_size,dtype=float)
    # calculate 3 sigma distance from centre
    x,y,z=np.ceil(3*sigma/voxel_size)
    distances=np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    gauss=-0.5*np.multiply(distances,distances)/sigma**2
    gauss3d=1
    for i in gauss: gauss3d = gauss3d*(np.exp(i)/np.sqrt(np.pi*2*sigma**2))
    print np.shape(gauss3d)
    return gauss3d

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
        tp=np.exp(-SSD*SSD/sigTSqrDouble)/tAxisLength

        #print kernel.shape,GausKern.shape,tp.shape
        coef=tp*GausKern
        coef=coef[...,None]
        out[center]=summing(summing(summing(  coef*kernel,axis=0  ),axis=0),axis=0)/summing(coef)

        it.iternext()
    return out

def TimeProfile_cl(data,sigTSqrDouble,GausKern,center,lenT):
    """Time profile clousness function. x and y should have shape= (1,1,1,len(time))"""
    summ=np.sum
    diff=data[center]-data
    SSD=summ(diff*diff,axis=-1)/lenT
    TclsKern=np.exp(-SSD*SSD/sigTSqrDouble)/lenT
    coeff=TclsKern*GausKern
    coeff=coeff[...,None]
    data_filtered=summ(summ(summ(data*coeff,0),0),0)/summ(coeff)

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
    size_half=int(size)/2
    #Calculating Gaussian kernel
    GausKern=gauss_kernel(size,sigG)
    sigTSqrDouble=float(sigT*sigT*2)
    center=(size_half,size_half,size_half)
    lenT=np.shape(img[0,0,0])

    img_filtered=np.zeros(np.shape(img))
    #making iterator which don't contain borders
    img_shp=np.shape(img[size:,size:,size:,0])

    summ=np.sum
    for x,y,z in np.ndindex(img_shp):
        kernel=img[x:x+size,y:y+size,z:z+size]


        diff=kernel[center]-kernel
        SSD=summ(diff*diff,axis=-1)/lenT
        TclsKern=np.exp(-SSD*SSD/sigTSqrDouble)/lenT
        coeff=TclsKern*GausKern
        coeff=coeff[...,None]
        img_filtered[x+size_half,y+size_half,z+size_half]=summ(summ(summ(kernel*coeff,0),0),0)/summ(coeff)


        #img_filtered[x+size_half,y+size_half,z+size_half]=TimeProfile_cl(kernel,sigTSqrDouble,GausKern,center,lenT)

    return img_filtered
def bilateralFunc(data,sigISqrDouble,GausKern,center=None):
    """ kernel should be  """
    diff=data[center]-data
    IclsKern=np.exp(-diff*diff/sigISqrDouble)
    coef=IclsKern*GausKern
    a = np.sum(data*coef)/np.sum(coef)
    return a



def bilateralFilter(img,voxel_size,sigg,sigi):
    """ 4d Bilateral exponential filter.
    image array, kernel size, distance SD, intensity SD
    """
    import ndbilateral
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