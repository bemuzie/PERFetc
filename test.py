__author__ = 'denis'
import numpy as np

a=np.arange(4*4*4*6).reshape(4,4,4,6)
for x in np.nditer(a, flags=['external_loop'],op_axes=[[0,1,2,-1]]):
    print x

