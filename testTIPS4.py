__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
from timeit import timeit

filelist=os.listdir('/media/WORK/_PERF/TVERYANOVICH Z.I. 14.07.1955/Nifti4d/')

adress='/media/WORK/_PERF/TVERYANOVICH Z.I. 14.07.1955/Nifti4d/'
adress_out='/media/WORK/_PERF/TVERYANOVICH Z.I. 14.07.1955/filtered/'
fileexct='GeneralBodyPerfusionDZEVANOVSKIISI30121947_3_FC17_QDS.nii'
for file in filelist[:]:
    img,hdr=image.loadnii(adress,file)
    print 'header', hdr
    shp=np.shape(img)
    img=image.crop(img,shp[0]/2,shp[1]/2,shp[2]/2,shp[2]/3)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
    print np.shape(img)
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
    tips=filters.std(img,13)
    image.savenii(tips,hdr,adress_out+'%s_%s_%s_%s.nii'%(file[:-4],size,sigmaG,sigmaT))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
