# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
from scipy import special, ndimage
from scipy.optimize import curve_fit,leastsq
from timeit import timeit
import filters



def func(data):
    for j in np.arange(3,3*3*3+3,3):
        print 'с ',j-3,'до',j, data[j-3:j]
    print
    return data[center]


a=np.arange(200*200*200*10).reshape(200,200,200,10)
print type(a)

b=filters.bilateralFilter(a,3,2,2)
