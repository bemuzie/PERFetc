__author__ = 'denis'
import numpy as np
import matplotlib.pyplot as plt

level=99
net=np.random.binomial(1,.5,(100000,level+1))
print np.sum(net,dtype=float)/(net.shape[0]*net.shape[1])
net=np.cumsum(net,-1)
print net.shape
net[net<5]=1
net[net>=5]=0
net=np.sum(net,1)
print net

hist=np.histogram(net,level,(0,level))
print hist

plt.plot(range(level),hist[0])
plt.show()

