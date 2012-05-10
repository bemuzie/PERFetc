__author__ = 'denis'
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import stats
from scipy import optimize as spopt
from curves import curves
from scipy.optimize import curve_fit,leastsq

#make time axis with nice time resolution
timestep=0.1
tc=np.arange(0,80,timestep)
#make time axis in moments of scaning/ 2 series:1st 11 scans in 2 sec,
# after 8 sec
# 6 scans in 4 sec
ts=np.append(np.arange(0,22,2), np.arange(30,54,4)).tolist()

# modeling concentrarion changing in aorta with gamma distribution pdf
class Compartment:
    def __init__(self,disttype,distpars,vol,inflow,time=tc,sertime=ts):
        """
        disttype - function of dynamic volumes distribution or vascular distribution
        distpars - list of distribution parameters
        vol - vascular/tissue volume ratio
        time - time axis
        sertime - time when scanning was performed
        inflow - input flow concentration
        """
        self.time=time
        self.pdf=disttype.pdf(time,*distpars)
        self.cdf=disttype.cdf
        self.rf=(1-disttype.cdf(time,*distpars)) / np.sum(1-disttype.cdf(time,*distpars))
        self.vol=vol
        self.inflow=inflow
        self.concentration=np.convolve(inflow,self.rf)
        self.outflow=np.convolve(inflow,self.concentration)
        #estimating visible concentration
        if not type(time) == list:
            print 'a'
            time=time.tolist()
        self.visibleconc=[ self.outflow[:len(time)] [time.index(i)] for i in sertime]

    def addnoise(self,sd):
        self.visibleconc+=np.random.normal(scale=sd,size=np.shape(self.visibleconc))

def throughnormal(time,mean,sigma,vol):
    curve=stats.norm.pdf(time,sigma,mean)
    return np.convolve(inflow, vol*curve/np.trapz(curve))

#injection
signal=np.zeros(len(tc))+20
signal[10:10/timestep]=400
#Concentration in aorta and recirculation
aorta=np.exp(-tc/1.5)*tc**3
arterialconc=aorta
#Concentration in tissue
tissue=Compartment(stats.norm,[20,4],0.1,arterialconc)
#Concentration in tumor
tumor=Compartment(stats.norm,[20,8],0.1,arterialconc)
#making ROI
ROIsize=(1000)
ROI=np.ones(ROIsize)[...,None]
tissue.visibleconc=tissue.visibleconc*ROI
print np.shape(tissue.visibleconc)
#adding noise

#Estimation of blood flow
            #spopt.curve_fit(throughnormal,ts,tumor.visibleconc,)
#Making graphs
zoom=5


plt.subplot(211)

plt.plot(tc,arterialconc,'k')
plt.plot(tc,tissue.concentration[:len(tc)],'r',
        tc,tumor.concentration[:len(tc)],'b')
#plt.plot(tc,zoom*fited,'--k')

plt.subplot(212)
"""
plt.plot(aorta.pdf)
plt.plot(recirculation.pdf)
plt.plot(tissue.pdf)
plt.plot(tumor.pdf)
"""
plt.plot(
    tc,tissue.rf[:len(tc)],'r-',
    tc,tumor.rf[:len(tc)],'b-'
        )

plt.show()


