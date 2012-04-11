
__author__ = 'ct'
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import networkx as nx
from scipy import ndimage


tc=np.linspace(0.00001,60,10000)
td=np.linspace(0.00001,60,17)

aorta_cont=stat.gamma.pdf(tc,9)*100
aort_discr=stat.gamma.pdf(td,1)

#Transit time distribution
m=50
s=5
ttdist=stat.norm.pdf(np.linspace(m-4*s,m+s*4,8*s/0.006),m,s)

tissue_concentration=ndimage.convolve1d(aorta_cont,ttdist,mode='constant')
print tissue_concentration
plt.subplot(311)
plt.plot(tc,aorta_cont)
plt.subplot(312)
plt.plot(np.linspace(m-4*s,m+s*4,8*s/0.006),ttdist)
plt.subplot(313)
plt.plot( np.linspace(0,60,len(tissue_concentration)), tissue_concentration)
plt.show()
