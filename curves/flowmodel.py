
__author__ = 'ct'
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstat

time=np.linspace(0,60,20)
timecont=np.arange(0,300,0.1)


concArt=spstat.gamma.pdf(timecont,3,3,2)
concArt_rec=spstat.gamma.pdf(timecont,10,3,2)
concArt2=concArt_rec+concArt

concArt[np.isnan(concArt)]=0

signal=np.zeros(len(timecont))
signal[800:1000]=2

sigma=10
ArivalTime=2


vascdist=spstat.norm.pdf(np.arange(-4*sigma,4*sigma,0.1),0,sigma)

vascdist=vascdist/np.sum(vascdist)

print vascdist, len(np.arange(-4*sigma,4*sigma,0.1))

concTissue=np.convolve(concArt2,vascdist,mode='same')
signalconv=np.convolve(signal,vascdist,mode='same')

print concArt
tacs=plt.subplot(1,2,1)
dist=plt.subplot(122)

tacs.plot(timecont,concArt2,'b')
tacs.plot(timecont,concArt,'r')
tacs.plot(timecont,concArt_rec,'k')
tacs.plot(np.arange(ArivalTime,ArivalTime+len(concTissue)*0.1,0.1),concTissue,'--')

dist.plot(timecont,signal)
dist.plot(timecont,signalconv)
plt.show()
