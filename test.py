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

sh=40
a=np.arange(sh*sh*sh*17).reshape(sh,sh,sh,17)


print (timeit("filters.bilateralFilter_t(a,5,2,2)", 'from __main__ import *', number = 1))
