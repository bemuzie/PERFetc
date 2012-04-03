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
    def addroi(self,coords,name,thick=0,invert=True):
        if invert==True:
            for i in range(3):
                coords[i]=np.shape(self.vol)[i]-coords[i]

        x,y,z=coords
        self.cords[name]=coords
        axial=np.average(self.vol[...,z-thick:z+thick+1],2)
        coronal=np.average(self.vol[x-thick:x+thick+1,:,:],0)
        print np.shape(coronal)
        sagital=np.average(self.vol[:,y-thick:y+thick+1,:],1)
        self.rois[name]=[np.rot90(i) for i in [axial,coronal,sagital]]


folder='/media/WORK/_PERF/TIKHEEV YU.V. 19.02.1935/Nifti4d/'
folderout='/media/WORK/_PERF/TIKHEEV YU.V. 19.02.1935/pics/'
native=Slides(folder+'AbdomenHCTNativeTIKHEEVYUV19021935s003a003.nii','Нативная')
pancreatic=Slides(folder+'2PhaseLiverTIKHEEVYUV19021935s010a003.nii','Панкреатическая')
portal=Slides(folder+'2PhaseLiverTIKHEEVYUV19021935s011a004.nii','Портальная')
delay=Slides(folder+'2PhaseLiverTIKHEEVYUV19021935s012a005.nii','Отсроченная(10 мин.)')

native.addroi([144,324,82],'pancr1',3)
pancreatic.addroi([150,328,79],'pancr1',3)
portal.addroi([150,323,363],'pancr1',3)
delay.addroi([145,325,83],'pancr1',3)

native.addroi([162,289,82],'pancr_normal',3)
pancreatic.addroi([168,282,81],'pancr_normal',3)
portal.addroi([170,277,363],'pancr_normal',3)
delay.addroi([160,282,86],'pancr_normal',3)


for img in [native,pancreatic,portal,delay]:
    print np.sort(img.vol[[slice(i-3,i+3) for i in img.cords['pancr1']]])
    print np.average(img.vol[[slice(i-3,i+3) for i in img.cords['pancr1']]])
    print np.average(img.vol[[slice(i-3,i+3) for i in img.cords['pancr_normal']]])
low=-100
high=140
for i in range(3):
    for vol in [native,pancreatic,portal,delay]:
        for name,roi in vol.rois.items():
            print type(roi)
            plt.imshow(roi[i],cmap='gray',clim=(low,high),
                aspect=vol.sratio[i],
                interpolation='bicubic')
            plt.savefig(folderout+'%s%s%s_win(%s_%s).png'%(name,vol.phase,i,low,high),facecolor='k')
        #plt.show()