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

tc=np.arange(0,60,0.1)
ts=tc[::20]

distribution=stats.norm.pdf(tc,10,1)
distribution2=stats.norm.pdf(ts,10,1)

inflow=stats.gamma.pdf(tc,10,3,3)
inflow2=stats.gamma.pdf(ts,10,3,3)

conc=np.convolve(distribution,inflow)
conc2=np.convolve(distribution2,inflow2)
conc/=np.trapz(conc)
conc2/=np.trapz(conc2)
a=fftpack.fft(inflow)
plt.subplot(211)
plt.plot(tc,distribution,
        ts,distribution2,'o-k'
)
plt.plot(tc,inflow,'r',
    ts,inflow2,'ok'
)
plt.plot(tc,conc[:len(tc)],'b',
    ts,conc2[:len(ts)],'ob')

plt.subplot(212)
plt.plot(range(len(a)),a.imag)
plt.show()




