__author__ = 'denest'
from filters import filters
from timeit import timeit
import ndbilateral
import matplotlib.pyplot as plt
import numpy as np
import image
from scipy import ndimage
import pstats, cProfile


#making test image
volume=image.testimg(size=(100,100,10,1),
                    pattern={0:1000,0.3:-50,.5:1000})
voxel_size=[1,1,4]
sigmaGauss=2
sigmaIntens=1

"""
print 'pure cython optimized:',(timeit('ndbilateral.bilateral_optimized(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))

print 'pure cython:',(timeit('ndbilateral.bilateral(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
'from __main__ import *',number=1))

print 'filters.bilateral3d:  ',(timeit('filters.bilateral3d(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
'from __main__ import *',number=1))

print 'pure cython:',(timeit('ndbilateral.bilateral(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))
print 'pure cython optimized:',(timeit('ndbilateral.bilateral_optimized(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))
"""
volumen=volume+np.random.normal(0,400,volume.shape)

"""
print volume.ndim
mn=.1
mx=1
nmr=10
for i in range(1,nmr):
    a=filters.gauss_kernel_3d(mn+i*(mx-mn)/float(nmr),flsize=[7,7,7])
    plt.subplot(1,nmr,i)
    print mn+i*(mx-mn)/float(nmr)
    plt.imshow(a[...,a.shape[2]/2],interpolation='nearest',cmap="gray")
"""

volumef=ndimage.generic_filter(volumen,filters.weightedaverage,size=(7,7,1,1),extra_keywords=dict(fsize=(7,7,1),sigmadif=200,gaussian=2,dSqrSigma=160000) )


plt.subplot(231)
plt.imshow(volume[...,volume.shape[2]/2,0],interpolation='nearest',cmap="gray")
plt.subplot(232)
plt.imshow(volumen[...,volumen.shape[2]/2,0],interpolation='nearest',cmap="gray")
plt.subplot(233)
plt.imshow(volumef[...,volume.shape[2]/2,0],interpolation='nearest',cmap="gray")
plt.subplot(234)
plt.imshow(filters.gauss_kernel_3d(2,flsize=[7,7,7])[:,:,3],interpolation='nearest',cmap="gray")
print filters.gauss_kernel_3d(2,flsize=[7,7,7])[:,:,3]

plt.show()