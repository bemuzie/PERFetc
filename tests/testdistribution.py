__author__ = 'denest'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

time=np.arange(0,80,0.01)

signalsum=np.zeros(time.shape)


"""
for i in time:
    tm=np.arange(0-i,60-i,0.01)
    tm[tm<0]=0
    if i<10:
        sig=np.exp(-tm)
        sig[sig>=1]=0
        signalsum+=sig
    if i%10==0:
        plt.plot(time,sig,'b')
"""
i=0
for m in np.arange(0,30,3):
    signal_duration=20
    dsignal=np.zeros(time.shape)
    dsignal[:signal_duration/0.01]=1
    #m=5
    s=1
    t=0
    i+=1
    print m
    rf=1-stats.gamma.cdf(time,m,t,s)

    signalsum2=np.convolve(dsignal,rf)
    signalsum2/=np.trapz(rf)


    plt.plot(time,signalsum2[:len(time)],color=(1-m/30.,0,m/30.,1))
    plt.plot(time,dsignal,'g')

plt.show()

