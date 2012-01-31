# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
from scipy import special, ndimage
from scipy.optimize import curve_fit,leastsq
from timeit import timeit

def func(bufer):
    return bufer[1]-bufer

a=np.arange(100)
print ndimage.generic_filter(a,func,size=3,)
print func(a)