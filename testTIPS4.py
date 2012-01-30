__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
import nibabel as nib
from timeit import timeit

filelist=os.listdir("/mnt/data/_PERF/YAVNIK/NifTi4d")
adress='/mnt/data/_PERF/YAVNIK/NifTi4d'
adress_out='/mnt/data/_PERF/YAVNIK/filtered/'
fileexct='GeneralBodyPerfusionDZEVANOVSKIISI30121947_3_FC17_QDS.nii'
croppart=8
for file in filelist[1:2]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)
    img=image.crop(img,-30+shp[0]/2,30+shp[1]/2,shp[2]/2,shp[2]/croppart)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
    print np.shape(img)
    tips=img
    size='crop'
    sigmaG=''
    sigmaT=''

    a=[400,700,1500]
    for i in a:
        size=9
        sigmaG=1.5
        sigmaT=i

        tips=tips4.convolve4d(img,size,sigmaG,sigmaT)

#    tips=filters.std(img,13)
    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s.nii'%(file[:-4],size,sigmaG,sigmaT,croppart))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
