__author__ = 'ct'
import numpy as np
import matplotlib.pyplot as plt

signal=np.array((9))
tiss=np.array((0,0,5),dtype=float)
tiss/=tiss.sum()
print tiss
print np.convolve(signal,tiss)