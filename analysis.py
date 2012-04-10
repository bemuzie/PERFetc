__author__ = 'denis'
import numpy as np
import matplotlib.pyplot as plt
import image
import matplotlib
import scipy.stats as stats
import curves

#time axis
tr=np.linspace(0.00001,60,1000)
#time when series are done
ts=np.append(np.arange(1,22,2),np.arange(22,60,4))
print ts
tissue=100*stats.lognorm.pdf(ts,0.25,0,40)
tumor=100*stats.lognorm.pdf(ts,.3,0,50)
plt.plot(ts,tissue,'b',ts,np.cumsum(tissue),'b')

for i in [tissue,tumor]:
    i+=np.random.normal(0,3,len(ts))
plt.plot(ts,tissue,'r',ts,np.cumsum(tissue),'r')
"""
mrx=np.zeros((512,512,len(ts)))
for i in np.ndindex(512,256):
    mrx[i]=stats.lognorm.pdf(ts,.3,1,40)


mrx+=np.random.normal(0,.01,(512,512,len(ts)))
#tissue=np.convolve(tissue,[1,1,1,2,1,1,1],mode='same')

for i in np.ndindex(512,512):
    curves.curves.fitcurve(mrx[i],ts)
plt.imshow(mrx[...,10])
"""

plt.show()
