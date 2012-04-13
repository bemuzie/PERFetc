__author__ = 'denis'
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import stats
from curves import curves
from scipy.optimize import curve_fit,leastsq

#make time axis with nice time resolution
tc=np.arange(0,80,0.1).tolist()
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
        self.vol=vol
        self.concentration=vol*disttype.pdf(time,*distpars)/np.trapz(disttype.pdf(time,*distpars))
        self.inflow=inflow
        self.outflow=np.convolve(inflow,self.concentration)
        self.visibleconc=[ self.outflow[:len(time)] [time.index(i)] for i in sertime]
    def addnoise(self,sd):
        self.visibleconc+=np.random.normal(scale=sd,size=len(self.visibleconc))

#injection
signal=np.zeros(len(tc))
signal[10:100]=4
#Concentration in aorta and recirculation
aorta=Compartment(stats.gamma,[3,2,2],1,signal)
recirculation=Compartment(stats.gamma,[3,10,8],1,aorta.outflow)
recirculation2=Compartment(stats.gamma,[3,10,20],1,recirculation.outflow)
arterialconc=aorta.outflow[:len(tc)]+recirculation.outflow[:len(tc)]+recirculation2.outflow[:len(tc)]
#Concentration in tissue
tissue=Compartment(stats.norm,[5,4],0.1,arterialconc)
#Concentration in tumor
tumor=Compartment(stats.norm,[6,4],0.1,arterialconc)

#adding noise
level=.01
tumor.addnoise(level)
tissue.addnoise(level)
arterialconc+=np.random.normal(scale=level,size=len(arterialconc))

#Estimation of blood flow
popt=curves.fitcurve(tissue.visibleconc[:-4],ts[:-4],initial=(1,6,3,1,0))
fited=curves.logn(tc,popt[0],popt[1],popt[2],popt[3],popt[4])

#Making graphs
plt.subplot(211)

plt.plot(tc,arterialconc,'b',ts,[ arterialconc[tc.index(i)] for i in ts],'bo-',
    tc,10*tissue.outflow[:len(tc)],'k',ts,10*tissue.visibleconc,'ko-',
    tc,10*tumor.outflow[:len(tc)],'r',ts,10*tumor.visibleconc,'ro-',
    tc,10*fited,'--k')
plt.subplot(212)
plt.plot(aorta.pdf)
plt.plot(recirculation.pdf)
plt.plot(tissue.pdf)
plt.plot(tumor.pdf)
print tc[0:1]
plt.show()


