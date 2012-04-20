# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
import filters.filters as filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes3d
from matplotlib import cm
from scipy import ndimage
from scipy import stats
from scipy import fftpack
ts=90 #time step

time=np.arange(0,80,0.1)
time_d=time[::ts]

aif=stats.gamma.pdf(time,2.2,10,7)
aif+=stats.gamma.pdf(time,8,20,7)*.5
aif*=1
aif_d=aif[::ts]

Ak=aif_d[None,::-1]*np.ones(len(aif_d))[:,None]
Ak=np.triu(Ak)


rf=1-stats.gamma.cdf(time,6,10,7)
rf_d=rf[::ts]

conc=np.convolve(rf,aif)
conc_d=np.convolve(rf_d,aif_d)
conc/=np.trapz(conc)
conc_d/=np.trapz(conc_d)


cj=np.zeros(len(time_d))
j=range(len(time_d))
K=aif_d

for i in j:
    cjK[]


print Ak
plt.subplot(211)
plt.plot(time,aif,'r',
         time_d,aif_d,'ro')

plt.plot(time,conc[:len(time)]*20,'b')
plt.plot(time_d,conc_d[:len(time_d)],'go',
        time_d,np.dot(Ak,rf_d),'ko')

plt.subplot(212)
plt.imshow(Ak)
plt.show()




