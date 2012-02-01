__author__ = 'ct'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image
import os

adress="/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/4dNifTi/filtered"
filelist=os.listdir(adress)


for file in filelist[:1]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)

imgplot=mpimg.imshow(img[...,40,3])

