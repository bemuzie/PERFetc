__author__ = 'denis'
import numpy as np
import smth
from collections import defaultdict

def gausskernel(sigma, pxsize=np.array((1,1,1))):
    fsize=1.5*sigma//pxsize
    print fsize
    grid_slices=[slice(-fsize[i]*pxsize[i],fsize[i]*pxsize[i]+pxsize[i],pxsize[i]) for i in range(3)]
    kernel_grid=np.ogrid[grid_slices[0],grid_slices[1],grid_slices[2]]
    print kernel_grid
    kernel_euclid=np.sqrt( np.sum( np.power(kernel_grid,2) ) )
    kernel_gauss=kernel_euclid#some calculations
    return kernel_gauss

def getvaldict(array):
    output=defaultdict(list)
    print array.size
    for i in range(array.size):
        output[array.flat[i]].append(i)
    print output.keys()

getvaldict(gausskernel(1))

