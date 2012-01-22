# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
from scipy import special
from scipy.optimize import curve_fit,leastsq
import matplotlib.pyplot as plt
import curves
from timeit import timeit


x,x_t=curves.samplet()
y_n=np.array([29.439678284182307,32.55495978552279,38.67024128686327,39.77747989276139,71.72386058981233,136.8391420911528,194.29490616621985,269.4959785522788,380.9571045576408,387.7372654155496,326.79088471849866,101.19839142091153,94.85254691689008,122.11260053619303,125.9195710455764,112.39946380697052,99.48793565683646])
initial=np.array([8,1.3,8,30,5],dtype=float)

popt=curves.fitcurve(y_n[:12],x[:12])
#x=np.linspace(0.001,10.,100)
print popt
#y_n=y+np.random.normal(scale=.01,size=len(y))
#popt,preg=curve_fit(gammapdf,x,y_n)
print initial
y_in=curves.gammapdf(x,initial)
y_f=curves.logn(x,popt[0],popt[1],popt[2],popt[3],popt[4])
y_f_t=curves.logn(x_t,popt[0],popt[1],popt[2],popt[3],popt[4])

plt.plot(x,y_n,'o',x,y_f,'--r',x,y_in,'-o',x_t,y_f_t,'r')
#print (timeit('curves.fitcurve_lsq(y_n,x)', 'from __main__ import *', number = 100))
#print (timeit('curves.fitcurve(y_n,x)', 'from __main__ import *', number = 100))
#print (timeit('curve_fit(curves.gammapdf,x[:11],y_n[:11],p0=[initial])', 'from __main__ import *', number = 100))
print curve_fit(curves.gammapdf_c,x,y_n)
plt.show()
