__author__ = 'ct'
"""
from image import dcm_parser
fromfolder='/media/WORK/_PERF/temp/270112_20120127_173903504/'
tofolder='/media/WORK/_PERF/'
dcm_parser(fromfolder,tofolder)
"""
import numpy as np
import image
import filters

img,hdr,mrx=image.loadnii('/media/63A0113C6D30EAE8/_CT/KOLTSOVA  N.N. 07.05.1950/Nifti4d/',\
    '2PhaseLiverKOLTSOVANN07051950s006a003.nii')
tips=img[55:439,161:428,84:328,None]
adress_out='/media/63A0113C6D30EAE8/_CT/KOLTSOVA  N.N. 07.05.1950/Nifti4d/'
size='crop'
sigmaG=''
sigmaT=''
shp=np.shape(tips)

a=[100]
for i in a:
    size=5
    sigmaG=1
    sigmaT=i


        #print (timeit('tips4.convolve4d(img,13,3,100)', 'from __main__ import *', number = 10))

    tips=filters.bilateralFilter(tips,size,sigmaG,sigmaT)


image.savenii(tips,mrx,adress_out+'%s_%s_%s_%s_x%sy%sz%s.nii'%\
                                  ('2PhaseLiverKOLTSOVANN07051950s006a003',size,sigmaG,sigmaT,shp[0],shp[1],shp[2]))




