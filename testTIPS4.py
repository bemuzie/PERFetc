__author__ = 'ct'
import tips4,image
import numpy as np
from timeit import timeit
img,hdr=image.loadnii('/media/63A0113C6D30EAE8/_PERF/YAVNIK/Nifti4d','GeneralBodyPerfusionYAVNIKGA12081948s008a001_FC18.nii')
img=image.crop(img,256,256,160,80)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
print np.shape(image)
dim_c=3
sigG=2

a=tips4.convolve4d(img,13,3,100)
image.savenii(img,hdr,'/media/63A0113C6D30EAE8/_PERF/YAVNIK/TIPS_FC18_13_3_100.nii')
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
