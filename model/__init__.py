__author__ = 'denest'

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
tres=10000
t=np.linspace(5000./tres,5000,tres)


a=np.zeros(tres)
b=np.zeros(tres)
s=np.zeros(tres)
s[5:50]=100



af=stats.gamma.pdf(t,1.5,5,2)
bf=stats.norm.pdf(t,10,1)
af/=np.sum(af)
bf/=np.sum(bf)
i=0


def plot_ab(t=t,a=a,b=b):
    plt.plot(t,a,'r',t,b,'b')
def conv_ab(a=a,af=af,b=b,bf=bf,s=s,i=i):
    i2=i+1
    print i2
    a2=np.convolve(af,b+s)[:tres]
    b2=np.convolve(bf,a2)[:tres]
    diffa=a-a2
    diffb=b-b2
    plt.plot(t,a,'r',t,b,'b')
    #print np.sum(diffa*diffa)
    #print np.sum(diffb*diffb)
    if np.sum(diffa*diffa)>0.01 or np.sum(diffb*diffb)>0.01:
        conv_ab(a=a2,af=af,b=b2,bf=bf,s=s,i=i2)


conv_ab(a=a,af=af,b=b,bf=bf,s=s,i=i)
plt.plot(t,s/5)
plt.plot(t,af*100,'--',t,bf*100,'--')
plt.show()
