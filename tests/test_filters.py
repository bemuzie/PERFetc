__author__ = 'denest'
from filters import filters
from timeit import timeit
import ndbilateral
import matplotlib.pyplot as plt
import numpy as np
import image

#making test image
volume=image.testimg(size=(80,80,80,1),
                    pattern={0:100,.5:50})
voxel_size=[1,1,1]
sigmaGauss=2
sigmaIntens=1
print 'filters.bilateral3d optimized:  ',(timeit('filters.bilateral3d(volume,%s,%s,%s,ndbilateral.bilatfunc_opt)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))

print 'filters.bilateral3d:  ',(timeit('filters.bilateral3d(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
'from __main__ import *',number=1))

print 'pure cython:',(timeit('ndbilateral.bilateral(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))
print 'pure cython optimized:',(timeit('ndbilateral.bilateral_optimized(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))
