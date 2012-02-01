__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
import nibabel as nib
from timeit import timeit

adress="/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/4dNifTi/"
filelist=os.listdir(adress)
adress_out='/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered/'
fileexct='GeneralBodyPerfusionDZEVANOVSKIISI30121947_3_FC17_QDS.nii'
croppart=4
for file in filelist[:1]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)

    img=image.crop(img,-30+shp[0]/2,60+shp[1]/2,shp[2]/2,shp[2]/croppart)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
    print np.shape(img),shp
    tips=img
    size='crop'
    sigmaG=''
    sigmaT=''

    a=[3000]
    for i in a:
        size=7
        sigmaG=1.5
        sigmaT=i

        tips=filters.tips4d_m(img,size,sigmaG,sigmaT)

#    tips=filters.std(img,13)
    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_TIPS.nii'%(file[:-4],size,sigmaG,sigmaT,croppart))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
