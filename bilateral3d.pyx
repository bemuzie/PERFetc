 #   Copyright 2012 Denis Nesterov
#   cireto@gmail.com
#
#   The software is licenced under BSD licence.
"""
  A cython implementation of bilateral filtering.
"""


import numpy as np
cimport numpy as np
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
def bilateralFunc(data,sigISqrDouble,GausKern,center=None):

    diff=data[center]-data
    IclsKern=np.exp(-diff*diff/sigISqrDouble)
    coef=IclsKern*GausKern
    return np.sum(data*coef)/np.sum(coef)

def bilateral(img,voxel_size,sigg,sigi,mpr):

     dimensions=np.ndim(img)
     if dimensions==3:
         img=img[...,np.newaxis]

     gaus_kern3d=gauss_kernel_3d(sigg, voxel_size)




     ks_x,ks_y,ks_z=np.asarray(np.shape(gaus_kern3d))/2

     print np.shape(img),ks_x,ks_y,ks_z

     gaus_kern=np.ravel(gaus_kern3d)
     center=len(gaus_kern)/2
     # calculate 2*sigma^2 of intensity closeness function out from loop
     sigISqrDouble=2*sigi**2
     #Closness function
     kwargs=dict(sigISqrDouble=sigISqrDouble,GausKern=gaus_kern,center=center)


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