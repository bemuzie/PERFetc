__author__ = 'ct'
import tips4,image
import numpy as np
from timeit import timeit
img,hdr=image.loadnii('/media/data/_PERF/Rogachevskiy','GeneralBodyPerfusionROGACHEVSKIJVF10031945.nii')
img=image.crop(img,256,256,160,20)
#img=np.arange(6*6*4*10).reshape(6,6,4,10)
print np.shape(image)
dim_c=3
sigG=2

a=tips4.convolve4d(img,3,1,10000000)
#image.savenii(a,hdr,'/media/data/_PERF/Rogachevskiy/TIPS.nii')
print (timeit('tips4.convolve4d(img,3,1,10000000)', 'from __main__ import *', number = 10))
