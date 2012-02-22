__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
import nibabel as nib
from timeit import timeit

adress="/media/WORK/_PERF/SILAGI A.L. 23.05.1958/Nifti4d/"
filelist=os.listdir(adress)
adress_out='/media/WORK/_PERF/SILAGI A.L. 23.05.1958/filtered/'
croppart=8

img,hdr, mrx=image.loadnii(adress,"GeneralBodyPerfusionSILAGIAL23051958s034a010_FC70_AIRD_05s.nii")

shp=np.shape(img)
x,y,z=[198,319,318]
img=image.crop(img[...,None],x,y,z,shp[2]/croppart,invert=False)
"""
img=np.arange(50*50*40*17).reshape(50,50,40,17)

print np.shape(img),shp
shp=np.shape(img)

size='crop'
sigmaG=''
sigmaT=''
"""

a=[1000]
for i in a:
    size=7
    sigmaG=1
    sigmaT=i


        #print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))

    tips=filters.bilateralFilter(img,[0.515,.515,.25],sigmaG,i)
    tips=filters.bilateralFilter(tips,[0.515,.515,.25],sigmaG,200)
    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_x%sy%sz%s.nii'%\
                                             ('FC70_AIRD_05s',size,sigmaG,sigmaT,shp[0],x,y,z))
#    tips=filters.std(img,13)

