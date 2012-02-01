__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
import nibabel as nib
from timeit import timeit

filelist=os.listdir("/media/data/_PERF/YAVNIK_GA/Nifti4d/")
adress='/media/data/_PERF/YAVNIK_GA/Nifti4d/'
adress_out='/media/data/_PERF/YAVNIK_GA/filtered/'
fileexct='GeneralBodyPerfusionDZEVANOVSKIISI30121947_3_FC17_QDS.nii'
croppart=8
for file in filelist[:]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)

    img=image.crop(img,-30+shp[0]/2,100+shp[1]/2,shp[2]/2,shp[2]/croppart)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
    print np.shape(img),shp
    tips=img
    size='crop'
    sigmaG=''
    sigmaT=''

    a=[50,80,120,150,200]
    for i in a:
        size=7
        sigmaG=1.5
        sigmaT=i

        tips=filters.bilateralFilter(img,size,sigmaG,sigmaT)

#    tips=filters.std(img,13)
    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_bilrl.nii'%(file[:-4],size,sigmaG,sigmaT,croppart))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
