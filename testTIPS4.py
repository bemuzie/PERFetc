__author__ = 'ct'
import tips4,image
import numpy as np
import filters
import os
import nibabel as nib
from timeit import timeit

adress="/media/63A0113C6D30EAE8/_PERF/CHUPROVA N.E. 13.10.1948/Nifti4d"
filelist=os.listdir(adress)
adress_out='/media/63A0113C6D30EAE8/_PERF/CHUPROVA N.E. 13.10.1948/filtered/'
croppart=8
for file in filelist[:1]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)

    img=image.crop(img,shp[0]/2,60+shp[1]/2,-20+shp[2]/2,20+shp[2]/croppart)
#img=np.arange(50*50*40*17).reshape(50,50,40,17)
    print np.shape(img),shp
    tips=img
    size='crop'
    sigmaG=''
    sigmaT=''

    a=[100]
    for i in a:
        size=3
        sigmaG=0.8
        sigmaT=i

        tips=filters.bilateralFilter(img,size,sigmaG,sigmaT)

#    tips=filters.std(img,13)
    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s.nii'%(file[:-4],size,sigmaG,sigmaT,shp[0]))
#print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))
