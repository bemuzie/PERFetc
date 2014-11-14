from scipy import stats 
import numpy as np
import matplotlib.pyplot as plt


s=np.arange(10)
a=np.arange(10)
lags = np.arange(10)
print lags
t = np.arange(10)[...,None].repeat(10,axis=1)
t = np.insert(t,[2,2],1000)
print t.shape,t

rc = 1-stats.gamma.cdf(t,1,s,a)


#plt.plot(t,rc)
#plt.show()
