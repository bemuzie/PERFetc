# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit,leastsq
from scipy import special as ssp
import matplotlib.pyplot as plt
from math import pi as pi

pipow=np.power(pi,0.5)*2
def logn(time,a=1,m=1,s=1,ts=1,b=1):
    """PDF of lognormal distribution
    0 t=time
    1 a=area under curve
    2 m=location
    3 s=scale
    4 ts - arrival time
    5 b=base level
    """
    t2=time-ts
    t2[t2<=0]=time[0]
    lognPdf=b+a*np.exp(-(np.log(t2)-m)*(np.log(t2)-m)/(2*s*s))/(t2 *s*pipow)
    #print('t1',time[0])
    return lognPdf
def gammapdf(t,coeffs):
    """
    coeffs:
    0=shape
    1=scale
    2=amplitude
    3=noise*background
    4=time step , time when lable arrive to point
    """
    t2=t-coeffs[4]
    t2[t2<=0]=t[0]
    pdf=coeffs[3]+coeffs[2]*pow(t2,coeffs[0]-1)*np.exp(-t2/coeffs[1]) / ssp.gamma(coeffs[0])*pow(coeffs[1],coeffs[0])
    return pdf
def gammapdf_c(t,sh,sc,amp,bg,ts):
    """
    coeffs:
    0 sh=shape
    1 sc=scale
    2 amp=amplitude
    3 bg=noise*background
    4 ts=time step , time when lable arrive to point
    """
    t2=t-ts
    t2[t2<=0]=t[0]
    pdf=bg+amp*pow(t2,sh-1)*np.exp(-t2/sc) / ssp.gamma(sh)*pow(sc,sh)
    return pdf
def passcurve_l(t,n,m,s,ts,tc,b,cont=True):
    if cont:
        return logn(t,n,m,s,ts,b), logn(tc,n,m,s,ts,b)
    return logn(t,n,m,s,ts,b)

def samplet(fl=11,fp=2.,sl=6,sp=4.,cont=True):
    #fl= lenth of first series fp - period of scans
    #sl - length of second series sp - period of sl scans
    #making real time array
    #tr - real time0 samples tc - continious time samples
    fl=float(fl)
    fp=float(fp)
    sl=float(sl)
    sp=float(sp)

    tr=(np.arange(fl))*fp
    tr[0]=0.0001
    tr_add=np.arange(sl)*sp+28
    tr=np.append(tr,tr_add)
    tc=np.linspace(np.min(tr),np.max(tr),100)
    if cont:
        return tr,tc
    return tr

def residuals(coeffs,data,t):
    return data-gammapdf(t,coeffs)

def fitcurve(data,time,initial=[],type='lgnorm'):
    #fitting curve function
    if type=='lgnorm':
        b=np.min(data)
        area=np.trapz(data-b,time)
        try:
            popt,pcov=curve_fit(logn,time,data,p0=(area,4,0.6,1,10))
        except RuntimeError:
            popt=[area,4,0.6,1,b]
            print('error')
            pass
    else:
        initial=np.array([8,1.3,8,30,2],dtype=float)
        try:
            popt,smth=curve_fit(gammapdf_c,time,data,p0=initial)
        except RuntimeError:
            print('error')
            popt=initial
            pass
    return popt

def fitcurve_lsq(data,time,func='gamma'):
    """Fitting curves with least squares method
    """
    initial=np.array([8,1.3,8,30,5],dtype=float)
    #func=gammapdf
    popt,smth=leastsq(residuals,initial,args=(data,time),maxfev=1500)
    return popt

def maxgrad(data):
    return np.max(np.gradient(data))

def modelfit(data,data_tc,time,ns,it=1,name='none'):
    """ Model noisy data and try to fit true curve 'it' times
    """
    grad_dif=np.array([])
    tc=np.linspace(np.min(time),np.max(time),100)
#fitting tissue curve
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

def FitArray(data,time):
    PerfPopt=np.zeros(data[:,:,:,0:5].shape) #making array storing values
    PerfVolume=np.zeros(data[:,:,:,0:2].shape)
    tr,tc=samplet(time[0],time[1],time[2],time[3]) # making time axis
    it = np.nditer (data[:,:,:,0], flags=['multi_index']) #iterator over volume axises
    size=float(np.prod(data[:,:,:,0].shape))
    i=0
    while not it.finished:
        popt=fitcurve(data[it.multi_index],tr) # fitting passage curve
        #PerfPopt[it.multi_index]=popt
        RealCurve=logn(tc,popt[0],popt[1],popt[2],popt[3],popt[4]) # making curve on bigger time resolution axis
        slope=maxgrad(RealCurve) # maximum slope calculation
        amplitude=np.max(RealCurve) # maximum amplitude calculation
        PerfVolume[it.multi_index,0]=slope
        PerfVolume[it.multi_index,1]=amplitude
        i+=1
        print 'посчитано', i,'вокселей из',size
        it.iternext()
    return PerfVolume


