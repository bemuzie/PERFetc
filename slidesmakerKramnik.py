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
        mrx=img.get_header().get_sform()
        self.sratio=[1,mrx[2,2]/mrx[1,1],mrx[2,2]/mrx[1,1]]
        self.rois=dict()
        self.cords=dict()
        self.thick=dict()
    def addroi(self,coords,name,thick=0,invert=True,axial=True,sagital=True,coronal=True):
        if invert==True:
            for i in range(3):
                coords[i]=np.shape(self.vol)[i]-coords[i]

        x,y,z=coords
        self.cords[name]=coords
        self.thick[name]=thick*2+1
        if axial:
            axial=np.average(self.vol[...,z-thick:z+thick+1],2)
            axial=np.rot90(axial)
        if coronal:
            coronal=np.rot90(np.average(self.vol[x-thick:x+thick+1,:,:],0))
        if sagital:
            sagital=np.rot90(np.average(self.vol[:,y-thick:y+thick+1,:],1))
        self.rois[name]=[i for i in [axial,coronal,sagital]]


folder='/media/63A0113C6D30EAE8/_PERF/KRAMNIK D.D. 02.01.1937/Nifti/parsed/'
folderout='/media/63A0113C6D30EAE8/_PERF/KRAMNIK D.D. 02.01.1937/pics/'

native1=Slides(folder[:-7]+'20120314_10332562AbdomenPelvisKRAMNIKDDs003a001.nii','Нативная')
arterial1=Slides(folder+'0.nii','Артериальная')
portal1=Slides(folder+'1.nii','Портальная')
delay1=Slides(folder+'2.nii','Венозная')
#adding perfusion volume
perf=slides(folder[:-7]+'20120412_121054GeneralBodyPerfusionKRAMNIKDD02011997s005a003.nii','Перфузия')




for i in range(2,15):
    native1.addroi([0,0,64+i*3],'tumor%s'%(i),3,sagital=False,coronal=False)
    arterial1.addroi([0,0,70+i*3],'tumor%s'%(i),3,sagital=False,coronal=False)
    portal1.addroi([0,0,65+i*3],'tumor%s'%(i),3,sagital=False,coronal=False)
    delay1.addroi([0,0,71+i*3],'tumor%s'%(i),3,sagital=False,coronal=False)


low=-75
high=175


for vol in [native1,arterial1,portal1,delay1]:
    for name,roi in vol.rois.items():
        plt.imshow(roi[0],cmap='gray',clim=(low,high),
            aspect=vol.sratio[0],
            interpolation='bicubic')
        plt.savefig(folderout+'%s%s%s_win(%s_%s)_th%s.png'%(name,vol.phase,i,low,high,vol.thick[name]),facecolor='k')