__author__ = 'ct'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import image
import os
import nibabel as nib

class Slides:
    def __init__(self,niifile,phase):
        self.phase=phase
        img=nib.load(niifile)
        self.vol=img.get_data()
        self.mrx=img.get_header().get_sform()
        self.rois=dict()
    def addroi(self,coords,name,thick=0):

        x,y,z=coords
        axial=np.average(self.vol[...,z-thick:z+thick+1],2)
        coronal=np.average(self.vol[x-thick:x+thick+1,:,:],0)
        print np.shape(coronal)
        sagital=np.average(self.vol[:,y-thick:y+thick+1,:],1)
        self.rois[name]=[np.rot90(i) for i in [axial,coronal,sagital]]


folder='/media/63A0113C6D30EAE8/_CT/KUPRIJANOVA_ I N/'

native=Slides(folder+'ABDOMENNATIVEKUPRIJANOVAINs003a003.nii','Нативная')

native.addroi([109,245,218],'Образование 1',5)



for i in range(3):
    plt.imshow(native.rois['Образование 1'][i],cmap='gray',clim=(-50,100),aspect=1,interpolation='bicubic')
    plt.savefig(folder+'output%s.png'%(i),facecolor='k')
    plt.show()