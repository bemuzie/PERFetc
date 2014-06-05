import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.integrate import quadrature

def tdc_create(input_conc,time):
	windows = np.arange(10)
	volumes = np.arange(0.1,1,0.1)

def run_sum(input_conc,time,window):
	cm=np.concatenate([np.ones(window),np.zeros(window-1)])/float(window)
	print cm
	return np.convolve(input_conc,cm)
def run_trapz(input_conc,time,window):
	o=np.zeros(input_conc.shape)
	scale = np.max(input_conc)
	for i in range(len(input_conc)):
		fr = (i-window>0) and i-window or 0
		o[i] = np.trapz(input_conc[fr:i])/window
	return o



if __name__ == "__main__":
	import matplotlib.pyplot as plt
	a = np.array([0,10,12,54,90,190,333,287,200,100,50,60,70,88,65,72,81,77,79])
	t = [10,12,14,16,18,20,22,24,36,40,44,48,52,60,70,80,100,120,140]
	print len(t)
	#plt.plot(run_sum(a,1,10))
	
	p = interp1d(t, a, kind='cubic')
	plt.plot(np.arange(np.min(t),np.max(t),.1),p(np.arange(np.min(t),np.max(t),.1)))
	#print quadrature(p,[1,2,3],[2,3,4])
	plt.plot(t,run_trapz(a,1,4),color='red')

	pan2=np.zeros(len(t))
	for i in range(len(pan2)):
		
		fr = (t[i]-20>np.min(t)) and t[i]-20 or np.min(t)
		print fr, t[i]
		print quadrature(p,fr,t[i])[0]
		pan2[i]=quadrature(p,fr,t[i])[0]/20
	print p(22)
	plt.plot(t,pan2,color='blue')
	"""
	curves_mrx = np.zeros([len(a),len(range(10))])
	print curves_mrx.shape
	for j in range(10):
		curves_mrx[:,j]=run_trapz(a,1,j)
	plt.plot(curves_mrx,color='red')
	plt.plot(curves_mrx*.2,color='blue')
	
	"""

	plt.plot(t,a)
	plt.show()

