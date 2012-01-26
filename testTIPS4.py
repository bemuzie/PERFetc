__author__ = 'ct'
import tips4,image
import numpy as np
from timeit import timeit


adress='/media/63A0113C6D30EAE8/_PERF/YAVNIK/Nifti4d'
file='GeneralBodyPerfusionYAVNIKGAs005a001_FC17reg.nii'
img,hdr=image.loadnii(adress,file)
img=image.crop(img,256,256,160,20)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
print np.shape(image)

a=np.linspace(100,1500,15)
for i in a:
    size=9
    sigmaG=1.8
    sigmaT=i

    tips=tips4.convolve4d(img,size,sigmaG,sigmaT)
    image.savenii(tips,hdr,'/media/63A0113C6D30EAE8/_PERF/YAVNIK/OLD_TIPS_%s_%s_%s_%s.nii'%(file[:-4],size,sigmaG,sigmaT))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
