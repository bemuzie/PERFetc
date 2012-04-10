# -*- coding: utf-8 -*-
from curves import curves
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

mrx=np.zeros((512,512))




roi1=slice(0,len(mrx)/2 )
roi2=slice(len(mrx)/2,-1)

mrx[roi1]=1
mrx +=np.random.normal(0,5,np.shape(mrx))

for i in np.arange(1,100):
    r1=slice( -i+len(mrx)/2 ,len(mrx)/2)
    r2=slice(len(mrx)/2,len(mrx)/2+i)
    a=stat.ttest_ind(np.ravel(mrx[r1,r1]),np.ravel(mrx[r2,r2]))

    print a,np.shape(mrx[r1,r1])

plt.imshow(mrx)


plt.show()