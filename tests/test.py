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
from scipy import optimize

timestep=0.1
tc=np.arange(0,80,timestep)
scantime=len(tc)

signal=np.zeros(len(tc))
s_duration=10
s_start=0
s_amp=5
signal[s_start/timestep : (s_start+s_duration)/timestep]=s_amp

AIF=tc**3 * np.exp(-tc/1.5)

def concest(pars,tc):
    """
    0-signal duration
    1-signal amplitude
    2-signal amp
    3-
    4-
    5-
    """

    print pars
    signal=np.zeros(len(tc))
    signal[: pars[0]/timestep]=pars[1]
    rf=1-stats.gamma.cdf(tc,*pars[2:])
    r=np.convolve(signal,rf/np.sum(rf))[:scantime]
    plt.plot(tc,r)
    return r

errfunc = lambda p,t,y:concest(p,t)-y
plt.subplot(211)

p0=[1,10,20,5,1]
p1,success=optimize.leastsq(errfunc,p0[:],args=(tc,AIF),maxfev=10000000)


AIF2=concest(p1,tc)



plt.plot(tc,AIF,
        tc,AIF2,'--')
plt.subplot(212)
plt.plot(tc,stats.gamma.pdf(tc,*p1[:2]))
plt.show()