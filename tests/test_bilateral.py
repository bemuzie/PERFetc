__author__ = 'ct'
import numpy as np
import timeit
import filters
import ndbilateral

img=np.ones((50,50,50,3))
contour1=[slice(10,-10) for i in [0,1,2]]
contour2=[slice(15,-15) for i in [0,1,2]]
contour3=[slice(20,-20) for i in [0,1,2]]

img[contour1]=5
img[contour2]=15
img[contour3]=10

img_n=img+np.random.normal(0,7,(50,50,50,1))

voxel_size=[1,1,1]
gauss_sigma=1
intensity_sigma=3
num=1

ROI=[[0,None],[0,None],[0,None],[0,None]]

t = timeit.Timer("filters.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma,mpr=ROI)",
    "from __main__ import *")
t2 = timeit.Timer("ndbilateral.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma,mpr=ROI)",
    "from __main__ import *")
print "Pure python function", t.timeit(num), "sec"
print "Cython function", t2.timeit(num), "sec"
