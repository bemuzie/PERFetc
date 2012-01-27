__author__ = 'ct'
import tips4,image
import numpy as np
from timeit import timeit


adress='/media/63A0113C6D30EAE8/_PERF/YAVNIK/Nifti4d/'
file='GeneralBodyPerfusionYAVNIKGA12081948s006a001_FC13.nii'
img,hdr=image.loadnii(adress,file)
img=image.crop(img,256,256,160,150)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
print np.shape(image)
tips=img
size='SD'
sigmaG='along'
sigmaT='Time'

"""
a=[900]
for i in a:
    size=7
    sigmaG=1
    sigmaT=i

    tips=tips4.convolve4d(img,size,sigmaG,sigmaT)
"""
tips=np.std(img,axis=3)
image.savenii(tips,hdr,'/media/63A0113C6D30EAE8/_PERF/YAVNIK/%s_%s_%s_%s.nii'%(file[:-4],size,sigmaG,sigmaT))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
