__author__ = 'denest'
from filters import filters
from timeit import timeit
from filters import ndbilateral
import matplotlib.pyplot as plt
import numpy as np
import image

#making test image
volume=image.testimg(size=(50,50,50,1),
                    pattern={0:100,.5:50})
voxel_size=[1,1,1]
sigmaGauss=1
sigmaIntens=2


print 'filters.ndbilateral:  ',(timeit('ndbilateral.bilateral(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
    'from __main__ import *',number=1))
print 'filters.bilateral:  ',(timeit('filters.bilateral(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
'from __main__ import *',number=1))
print 'filters.bilateral3d:  ',(timeit('filters.bilateral3d(volume,%s,%s,%s)'%(voxel_size,sigmaGauss,sigmaIntens),
'from __main__ import *',number=1))
