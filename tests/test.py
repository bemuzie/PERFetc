# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
import filters.filters as filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes3d
from matplotlib import cm
from scipy import ndimage
from scipy import stats
from scipy import fftpack
ts=90 #time step

time=np.arange(0,80,0.1)
time_d=time[::ts]

for i in np.arange(1,3,0.1):
    aif=np.power(time,3)*np.exp(-time*i)
    plt.plot(time,aif)

plt.show()