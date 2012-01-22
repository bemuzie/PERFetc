__author__ = 'denis'
import numpy as np
from scipy import special
from scipy.optimize import curve_fit,leastsq
import matplotlib.pyplot as plt
import curves

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
    pdf=coeffs[3]+coeffs[2]*pow(t2,coeffs[0]-1)*np.exp(-t2/coeffs[1]) / special.gamma(coeffs[0])*pow(coeffs[1],coeffs[0])
    return pdf
initial=np.array([8,1.3,8,30,5],dtype=float)
def residuals(coeffs,data,t):
    return data-gammapdf(t,coeffs)
x,x_t=curves.samplet()
print x
y_n=np.array([29.439678284182307,32.55495978552279,38.67024128686327,39.77747989276139,71.72386058981233,136.8391420911528,194.29490616621985,269.4959785522788,380.9571045576408,387.7372654155496,326.79088471849866,101.19839142091153,94.85254691689008,122.11260053619303,125.9195710455764,112.39946380697052,99.48793565683646])

popt,smth=leastsq(residuals,initial,args=(y_n[:11],x[:11]),maxfev=10000)
#x=np.linspace(0.001,10.,100)
print popt
#y_n=y+np.random.normal(scale=.01,size=len(y))
#popt,preg=curve_fit(gammapdf,x,y_n)
print initial
y_in=gammapdf(x,initial)
y_f=gammapdf(x,popt)
plt.plot(x,y_n,'o',x,y_f,'--r',x,y_in,'-o')

plt.show()
