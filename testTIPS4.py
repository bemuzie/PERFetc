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
croppart=4
for file in filelist[3:4]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)
    x,y,z=[265,348,198]

    img=image.crop(img,x,y,z,-20+shp[2]/croppart,invert=False)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)

    print np.shape(img),shp
    shp=np.shape(img)
    tips=img
    size='crop'
    sigmaG=''
    sigmaT=''

    a=[1000]
    for i in a:
        size=5
        sigmaG=0.8
        sigmaT=i

        tips=filters.tips4d_m(img,size,sigmaG,sigmaT)
        tips=filters.bilateralFilter(tips,5,1,100)
#    tips=filters.std(img,13)

    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_x%sy%sz%s.nii'%(file[48:-4],size,sigmaG,sigmaT,shp[0],x,y,z))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
