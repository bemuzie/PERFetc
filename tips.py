# -*- coding: utf-8 -*-
"""
#TIPS filter described in "TIPS bilateral noise reduction
in 4D CT perfusion scans produces high-quality cerebral blood flow maps"
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import signal
def create(x,y,z,t,obj=2):
#create test array
	dim=np.array([x,y,z])
	wt = np.ones(dim,dtype=float)
	wt[obj:-obj,obj:-obj,obj:-obj] = wt[obj:-obj,obj:-obj,obj:-obj]*2
	tm = np.arange(1,t+1)
	data = wt[:,:,:,np.newaxis]*tm[np.newaxis,np.newaxis,np.newaxis,:]
	return data

def ssd (array, sigma=1):
#SSD across time
	dataext = array[:,:,:,:,np.newaxis]
	difdata = (dataext-np.swapaxes(dataext,3,4))
	difdata *= difdata
	ssd = np.sum(difdata,axis=-1)/len(difdata[1,1,1,1])
#TIPS
	return np.exp(-0.5*ssd/sigma*ssd/sigma)

def t_weighted_ar(array, sigma=1):
#make array weighted with TIPS	
	return ssd(array,sigma)*array

def g_kernel (x,y=[],z=[],sigma=1,out='xyz',norm=True):
#make Gaussian kernel	
	if y == z == []:
		y,z=x,x
#one dimmensional kernels	
	xkernel = np.exp(-0.5*(((np.arange(1,x+1)-ceil(x/2.)) /sigma)**2))
	ykernel = np.exp(-0.5*(((np.arange(1,y+1)-ceil(y/2.)) /sigma)**2))
	zkernel = np.exp(-0.5*(((np.arange(1,z+1)-ceil(z/2.)) /sigma)**2))
#four dimmesional
	xkernel = xkernel[:,np.newaxis,np.newaxis,np.newaxis]
	ykernel = ykernel[np.newaxis,:,np.newaxis,np.newaxis]
	zkernel = zkernel[np.newaxis,np.newaxis,:,np.newaxis]
#normalisation
	if norm == True:
		xkernel /= np.sum(xkernel)
		ykernel /= np.sum(ykernel)
		zkernel /= np.sum(zkernel)

	if out == 'xyz':
		return xkernel*ykernel*zkernel
	if out == 'x':
		return xkernel
	if out == 'y':
		return ykernel
	if out == 'z':
		return zkernel


def tips_filter (array,sigmaT=1,x=3,y=[],z=[],sigma=1,norm = True):
# TIPS filter 
	data_t=t_weighted_ar(array, sigmaT)
	data_t=signal.convolve(data_t,g_kernel(x=x,y=y,z=z,sigma=sigma,out='x'), 'same')
	data_t=signal.convolve(data_t,g_kernel(x=x,y=y,z=z,sigma=sigma, out='y'), 'same')
	data_t=signal.convolve(data_t,g_kernel(x=x,y=y,z=z,sigma=sigma,out='z'), 'same')
	if norm == True:
		tipsar=ssd(array, sigmaT)
		tipsar=signal.convolve(tipsar,g_kernel(x=x,y=y,z=z,sigma=sigma,out='x'), 'same')
		tipsar=signal.convolve(tipsar,g_kernel(x=x,y=y,z=z,sigma=sigma, out='y'), 'same')
		tipsar=signal.convolve(tipsar,g_kernel(x=x,y=y,z=z,sigma=sigma,out='z'), 'same')
		data_t /= tipsar
	return data_t
