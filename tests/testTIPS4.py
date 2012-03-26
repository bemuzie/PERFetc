__author__ = 'ct'
import  image
import numpy as np
import os
import ndbilateral

adress="/media/63A0113C6D30EAE8/_PERF/YAVNIK  G.A. 12.08.1948/20120321_509/Nifti4d/"
filelist=os.listdir(adress)
adress_out='/media/63A0113C6D30EAE8/_PERF/YAVNIK  G.A. 12.08.1948/20120321_509/filtered/'
croppart=4

img,hdr, mrx=image.loadnii(adress,"GeneralBodyPerfusionYAVNIKGA12081948s005a003.nii")

shp=np.shape(img)
print shp
x,y,z=[235,309,140]
img=image.crop(img[...,],x,y,z,100,invert=False)
print 'crop completed'
print np.shape(img)
print type(img[1,1,1,0])
img=img.astype('float64')
"""
img=np.arange(50*50*40*17).reshape(50,50,40,17)

print np.shape(img),shp
shp=np.shape(img)

size='crop'
sigmaG=''
sigmaT=''
"""
print type(img[1,1,1,0])

a=[10]
for i in a:
    size=0.4
    sigmaG=0.7
    sigmaT=i


        #print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))

    tips=ndbilateral.bilateral(img[...],[0.78,.78,.5],sigmaG,i)
    image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_crp%s_x%sy%sz%s.nii'%\
                                             ('FC17_QDS_1',size,sigmaG,sigmaT,shp[0],x,y,z))
    image.convert(tips,adress_out+'parsed/')
#    tips=filters.std(img,13)

