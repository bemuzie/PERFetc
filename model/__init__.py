__author__ = 'denest'

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
tres=500
t=np.linspace(100./tres,100,tres)


a=np.zeros(tres)
b=np.zeros(tres)
s=np.zeros(tres)
s[5:50]=100



af=stats.gamma.pdf(t,2,5,2)
bf=stats.norm.pdf(t,20,6)
af*=t[0]
bf*=t[0]
print np.sum(af)


def plot_ab(t=t,a=a,b=b):
    plt.plot(t,a,'r',t,b,'b')
def conv_ab(a=a,af=af,b=b,bf=bf,s=s):
    a2=np.convolve(af,b+s)
    b2=np.convolve(bf,a)
    return a2[:tres],b2[:tres]

for i in range(50):
    plot_ab(t,a,b)
    a,b=conv_ab(a=a,af=af,b=b,bf=bf,s=s)

plt.plot(t,s/5)
plt.plot(t,af*100,'--',t,bf*100,'--')
plt.show()
