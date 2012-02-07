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
croppart=4
for file in filelist[:]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)
    x,y,z=[240,297,203]

    img=image.crop(img,x,y,z,shp[2]/croppart,invert=False)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)

    print np.shape(img),shp
    shp=np.shape(img)
    tips=img
    size='crop'
    sigmaG=''
    sigmaT=''

    a=[50,200]
    for i in a:
        size=3
        sigmaG=0.8
        sigmaT=i


        tips=filters.bilateralFilter(tips,size,sigmaG,i)
#    tips=filters.std(img,13)

    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_x%sy%sz%s.nii'%(file[48:-4],size,sigmaG,sigmaT,shp[0],x,y,z))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
