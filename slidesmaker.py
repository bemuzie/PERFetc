__author__ = 'ct'
# -*- coding: utf-8 -*-
#Ivanova
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
    def addroi(self,coords,name,thick=0,invert=True):
        if invert==True:
            for i in range(3):
                coords[i]=np.shape(self.vol)[i]-coords[i]

        x,y,z=coords
        self.cords[name]=coords
        self.thick[name]=thick*2+1
        axial=np.average(self.vol[...,z-thick:z+thick+1],2)
        coronal=np.average(self.vol[x-thick:x+thick+1,:,:],0)
        print np.shape(coronal)
        sagital=np.average(self.vol[:,y-thick:y+thick+1,:],1)
        self.rois[name]=[np.rot90(i) for i in [axial,coronal,sagital]]


folder='/media/63A0113C6D30EAE8/_CT/IVANOVA N.A. 03.07.1954/Nifti/'
folderout='/media/63A0113C6D30EAE8/_CT/IVANOVA N.A. 03.07.1954/pics/'

native1=Slides(folder+'20110215_132420ChestAbdomenHCTRecon3mmIVANOVANA03071954s011a004.nii','Нативная0215')
arterial1=Slides(folder+'20110215_1324203PhaseLiverPORTAEIVANOVANA03071954s012a003.nii','Артериальная0215')
portal1=Slides(folder+'20110215_1324203PhaseLiverPORTAEIVANOVANA03071954s014a004.nii','Портальная0215')
delay1=Slides(folder+'20110215_1324203PhaseLiverPORTAEIVANOVANA03071954s016a005.nii','Отсроченная0215')

native2=Slides(folder+'20111124_122115AbdomenHCTNativeIVANOVANA03071954s003a003.nii','Нативная1124')
arterial2=Slides(folder+'20111124_1221152PhaseKidneyIVANOVANA03071954s007a003.nii','Артериальная1124')
portal2=Slides(folder+'20111124_1221152PhaseKidneyIVANOVANA03071954s008a004.nii','Портальная1124')


native1.addroi([401,166,415],'liver1',3)
arterial1.addroi([399,169,508],'liver1',3)
portal1.addroi([391,172,998],'liver1',3)
delay1.addroi([398,163,511],'liver1',3)
native2.addroi([408,224,107],'liver1',1)
arterial2.addroi([408,225,52],'liver1',1)
portal2.addroi([409,222,107],'liver1',1)

native1.addroi([401,166,415],'liver2',3)
arterial1.addroi([399,169,508],'liver2',3)
portal1.addroi([391,172,998],'liver2',3)
delay1.addroi([398,163,511],'liver2',3)
native2.addroi([408,224,107],'liver2',1)
arterial2.addroi([408,225,52],'liver2',1)
portal2.addroi([409,222,107],'liver2',1)

low=-75
high=175
for i in range(3):
    for vol in [native1,arterial1,portal1,delay1,native2,arterial2,portal2]:
        for name,roi in vol.rois.items():
            print type(roi)
            plt.imshow(roi[i],cmap='gray',clim=(low,high),
                aspect=vol.sratio[i],
                interpolation='bicubic')
            plt.savefig(folderout+'%s%s%s_win(%s_%s)_th%s.png'%(name,vol.phase,i,low,high,vol.thick[name]),facecolor='k')
        #plt.show()