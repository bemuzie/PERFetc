__author__ = 'ct'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import image
import os
import nibabel as nib
from scipy import ndimage

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
        axialim=np.array
        coronalim=np.array
        sagitalim=np.array
        if axial:
            axialim=np.average(self.vol[...,z-thick:z+thick+1],2)
            axialim=np.rot90(axialim)
        if coronal:
            coronalim=np.rot90(np.average(self.vol[x-thick:x+thick+1,:,:],0))
        if sagital:
            sagitalim=np.rot90(np.average(self.vol[:,y-thick:y+thick+1,:],1))
        self.rois[name]=[i for i in [axialim,coronalim,sagitalim]]

recon='AIRD13'
folder='/media/WORK/_PERF/KRAMNIK D.D. 02.01.1937/Nifti/parsed%s/'%recon
folder2='/media/WORK/_PERF/KRAMNIK D.D. 02.01.1937/Nifti/'
folderout='/media/WORK/_PERF/KRAMNIK D.D. 02.01.1937/pics/'
"""
native1=Slides(folder[:-7]+'20120314_10332562AbdomenPelvisKRAMNIKDDs003a001.nii','Нативная')
arterial1=Slides(folder+'0.nii','Артериальная')
portal1=Slides(folder+'1.nii','Портальная')
delay1=Slides(folder+'2.nii','Венозная')
"""
#adding perfusion volume
perf=dict()
"""
for sernum in range(3,20):
    perf[sernum]=Slides(folder+'20120412_121054GeneralBodyPerfusionKRAMNIKDD02011997s005a0%02d.nii'%sernum,
                        'Перфузия%s'%sernum)
    print sernum
"""
for sernum in range(0,17):
    perf[sernum]=Slides(folder+'%s.nii'%sernum,
        'Перфузия%s'%sernum)
    print sernum

slidecords={
    3:[0,0,209],
    4:[0,0,211],
    5:[0,0,215],
    6:[0,0,217],
    7:[0,0,221],
    8:[0,0,220],
    9:[0,0,223],
    10:[0,0,221],
    11:[0,0,221],
    12:[0,0,219],
    13:[0,0,224],
    14:[0,0,209],
    15:[0,0,209],
    16:[0,0,207],
    17:[0,0,207],
    18:[0,0,210],
    19:[0,0,208]
}
num=range(-8,5)
"""
for i in num:
    for ser,cord in slidecords.items()[7:8]:
        x,y,z=cord
        perf[ser].addroi([x,y,z+6*i],'perf%02d_%s'%(ser,i-num[0]),2,invert=False,sagital=False,coronal=False)
"""
for i in num:
    for ser,cord in slidecords.items()[7:8]:
        x,y,z=cord
        perf[ser].addroi([x,y,z+7*i],'%s'%(13-i+num[0]),7,invert=False,sagital=False,coronal=False)

"""
for i in range(2,4):
    native1.addroi([0,0,64+i*3],'tumor%s'%(i),2,sagital=False,coronal=False)
    arterial1.addroi([0,0,70+i*3],'tumor%s'%(i),2,sagital=False,coronal=False)
    portal1.addroi([0,0,65+i*3],'tumor%s'%(i),2,sagital=False,coronal=False)
    delay1.addroi([0,0,71+i*3],'tumor%s'%(i),2,sagital=False,coronal=False)
"""

low=-100
high=200

"""
for vol in [native1,arterial1,portal1,delay1]:
    for name,roi in vol.rois.items():
        plt.imshow(roi[0],cmap='gray',clim=(low,high),
                    aspect=vol.sratio[0],
                    interpolation='bicubic')
        plt.savefig(folderout+'%s%s%s_win(%s_%s)_th%s.png'%(name,vol.phase,i,low,high,vol.thick[name]),facecolor='k')
"""

"""
for ser,vol in perf.items():
    for name,roi in vol.rois.items():
        plt.imshow(roi[0],cmap='gray',clim=(low,high),
                    aspect=vol.sratio[0],
                    interpolation='bicubic')
        plt.savefig(folderout+'%s%s_win(%s_%s)_th%s_%s.png'%(name,vol.phase,low,high,vol.thick[name],recon),facecolor='k')

"""
for ser,vol in perf.items():
    for name,roi in vol.rois.items():
        plt.imshow(roi[0],cmap='gray',clim=(low,high),
            aspect=vol.sratio[0],
            interpolation='bicubic')
        plt.savefig(folderout+'image%s_.png'%(name),facecolor='k')
