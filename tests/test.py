# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
import filters.filters as filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes3d
from matplotlib import cm
from scipy import ndimage as ndimage

plt.imshow(filters.gauss_kernel_3d(1,[.7,.7,.5])[...,0],interpolation='nearest')
plt.show()
