__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
import nibabel as nib
from timeit import timeit

adress="/media/63A0113C6D30EAE8/_PERF/BOZHICH  N.I. 15.04.1950/Nifti4d/"
filelist=os.listdir(adress)
adress_out='/media/63A0113C6D30EAE8/_PERF/BOZHICH  N.I. 15.04.1950/filtered/'
croppart=2
for file in filelist[:]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)
    x,y,z=[256,256,160]

    #img=image.crop(img,x,y,z,shp[2]/croppart,invert=False)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)

    print np.shape(img),shp
    shp=np.shape(img)

    size='crop'
    sigmaG=''
    sigmaT=''

    a=[150]
    for i in a:
        size=3
        sigmaG=0.8
        sigmaT=i

        for times in [0,4,10]:
            #print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))

            tips=filters.bilateralFilter(img[...,times:times+1],size,sigmaG,i)
            image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_x%sy%sz%s_ts%s.nii'%\
                                                     (file[48:-4],size,sigmaG,sigmaT,shp[0],x,y,z,times))
        #    tips=filters.std(img,13)

