# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
from scipy import special, ndimage
from scipy.optimize import curve_fit,leastsq
from timeit import timeit
import filters



def func(data):
    b=np.array([])
    shp=np.shape(data[...,0])
    for i in np.ndindex(shp):
        b=np.append(b,data[i])

    print 'reshape'
    b=np.reshape(b,np.shape(data))
    print b==data

def func2(data):
    shp=np.shape(data)
    b=np.ones(shp)
    shp=np.shape(data[...,0])
    for i in np.ndindex(shp):
        b[i]=data[i]

    print b==data

sh=100

a=np.random.normal(50,20,size=sh*sh*sh*17).reshape(sh,sh,sh,17)



print (timeit("filters.tips4d(a,5,2,2)", 'from __main__ import *', number = 1))
#print (timeit("func2(a)", 'from __main__ import *', number = 1))