__author__ = 'denest'
from filters import filters
from timeit import timeit
import ndbilateral
import matplotlib.pyplot as plt
import numpy as np
import image
import pstats, cProfile
import smth

print 'pure cython optimized:',(timeit('smth.sumnum(1,1,100000000)',
    'from __main__ import *',number=1000))