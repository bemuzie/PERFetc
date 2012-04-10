# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
import filters.filters as filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes3d
from matplotlib import cm
from scipy import ndimage as ndimage
from scipy import stats as ss

class Tissue:
    def __init__(self,size,perf):
        """ size - list of sizes in each dimmension"""
        self.size=size
        self.tissue=np.zeros(size)
        self.perf=ss.lognorm.pdf()


