__author__ = 'ct'
import numpy as np
import timeit
import filters
import ndbilateral
import matplotlib.pyplot as plt

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

print img_n.shape
a=filters.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)
b=ndbilateral.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)
"""
print a-b
t = timeit.Timer("filters.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)",
    "from __main__ import *")
t2 = timeit.Timer("ndbilateral.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)",
    "from __main__ import *")
print "Pure python function", t.timeit(num), "sec"
print "Cython function", t2.timeit(num), "sec"
"""
sp1=plt.subplot(311)
sp2=plt.subplot(312)
sp3=plt.subplot(313)
print a == img_n

sp1.set_title('noisy')
sp2.set_title('python')
sp3.set_title('cython')

sp1.imshow(img_n[...,25,2])

sp2.imshow(a[...,25,2])
sp3.imshow(b[...,25,2])

plt.show()

