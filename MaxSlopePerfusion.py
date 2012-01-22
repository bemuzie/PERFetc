__author__ = 'denis'
import curves, image
import numpy as np
folder='/media/data/_PERF/Rogachevskiy'
name='GeneralBodyPerfusionROGACHEVSKIJVF10031945.nii'
header,img=image.loadnii(folder,name)
#img=img[:,:,:100,0:1]
ish=img.shape
img=image.crop(img,ish[0]-228,ish[1]-272,ish[2]-184,20)
time=(11,2,6,4)
PerfMap = curves.FitArray(img,time)
print PerfMap.shape

#PerfMap = PerfMap[:,:,:,0]/np.max(PerfMap[:,:,:,1])
print PerfMap.shape
image.savenii(PerfMap,header,folder+'/PerfMap.nii')

#image.savenii(img,header,folder+'/PerfMap.nii')