# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gc
from math import pi as pi

gc.disable()

pipow=np.power(pi,0.5)*2
def logn(time,a=1,m=1,s=1,ts=1,b=1):
    #PDF of lognormal distribution
    #t=time a=area under curve m=location s=scale ts - arrival time b=base level
    t2=time-ts
    t2[t2<=0]=time[0]
    lognPdf=b+a*np.exp(-(np.log(t2)-m)*(np.log(t2)-m)/(2*s*s))/(t2 *s*pipow)
    #print('t1',time[0])
    return lognPdf

def passcurve_l(t,n,m,s,ts,tc,b,cont=True):
    if cont:
        return logn(t,n,m,s,ts,b), logn(tc,n,m,s,ts,b)
    return logn(t,n,m,s,ts,b)
#def passcurve_g
#making pass curve by gamma.pdf
#	if type == 'gamma':
#		return ss.gamma.pdf(t,n,m,s)*400

def samplet(fl=11,fp=2.,sl=6,sp=4.,cont=True):
    #fl= lenth of first series fp - period of scans 
    #sl - length of second series sp - period of sl scans
    #making real time array
    #tr - real time samples tc - continious time samples
    tr=(np.arange(fl))*fp
    tr[0]=0.0001
    tr_add=np.arange(sl)*sp+28
    tr=np.append(tr,tr_add)
    tc=np.linspace(np.min(tr),np.max(tr),100)
    if cont:
        return tr,tc
    return tr

def fitcurve(time,data):
    #fitting curve function
    b=np.min(data)
    area=np.trapz(data-b,time)
    try:
        popt,pcov=curve_fit(logn,time,data,p0=(area,4,0.6,1,b))
    except RuntimeError:
        popt=[area,4,0.6,1,b]
        print('error')
        pass
    #print ('area',area,'base',b)
    
    return popt

def maxgrad(data):
    return np.max(np.gradient(data))
#making time steps, tr=real samples, tc = continuous samples
tr,tc = samplet()

def modelfit(data,data_tc,time,ns,it=1,name='none'):
    grad_dif=np.array([])
    tc=np.linspace(np.min(time),np.max(time),100)
#fitting tissue curve
    gc.disable()
    for i in np.arange(it):
#adding noise to real time samples curve
        data_n=data+np.random.normal(scale=ns,size=len(data))
#        if name == 'art':
#            plt.plot(time,data_n,'.k')
#        plt.plot(time,data_n,'_m')
#fitting curve
        data_p=fitcurve(time,data_n)
#making fitted curves
#        data_f=logn(tr,data_p[0],data_p[1],data_p[2],data_p[3],data_p[4])
        data_f_tc=logn(tc,data_p[0],data_p[1],data_p[2],data_p[3],data_p[4])
        plt.plot(tc,data_f_tc,'--b')
#calculating maximum gradient
        if name=='tiss':
            prnt_true=maxgrad(data_tc)
            prnt_fit=maxgrad(data_f_tc)
        if name == 'art':
            prnt_true=np.max(data_tc)
            prnt_fit=np.max(data_f_tc)
        grad_dif= np.append(grad_dif,abs(prnt_true-prnt_fit))
        print 100*i/float(it)
    print(name,'average',np.average(grad_dif),'SD',np.std(grad_dif))
    print('max true',prnt_true,'max fit',prnt_fit)
    return data_n

#making passage curve
tiss,tiss_tc= passcurve_l(tr,3000.,3.,0.6,ts=10.,tc=tc,b=50.)
tissS,tissS_tc= passcurve_l(tr,1000.,3.5,0.6,ts=20.,tc=tc,b=0.)
tissR_tc=tiss_tc+tissS_tc
tissR=tiss+tissS
artflow,artflow_tc=passcurve_l(tr,4000.,2.,0.75,ts=3.,tc=tc,b=40.)

tissn=modelfit(tissR,tiss_tc,tr,it=1000,ns=5,name='tiss')
artflown=modelfit(artflow,artflow_tc,tr,it=1000,ns=10,name='art')


#plot passage curves
plt.plot(tc,tiss_tc,'-r',tr,tissn,'o-g',tc,tissR_tc,'r')

plt.plot(tc,artflow_tc,'-r',tr,artflown,'o-g')
#print(maxgrad,maxgrad_f)
#print (popt)
plt.show()

