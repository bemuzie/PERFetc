__author__ = 'ct'
import image
import os
import nibabel as nib
from scipy import ndimage as ndimage
import numpy as np
import curves
import matplotlib.pyplot as plt

adress='/media/63A0113C6D30EAE8/_PERF/VLASYUK E.V. 07.10.1992/Nifti4d/'
volume='GeneralBodyPerfusionVLASYUKEV07101992s004a001.nii'

img,hdr,mrx=image.loadnii(adress,volume)
cntr=[318,255,155]
sideratio=mrx[2,2]/mrx[1,1]
panc=curves.Roi(img,cntr,10,filtr=True,voxsize=[0.6,0.6,.5],sigg=0.8,sigi=100,phase=6,rotation=1)
print np.shape(panc.sliceAx)

sp1=plt.subplot(131)
sp1.imshow(panc.sliceAx,cmap='gray',clim=(-200,300))
sp1=plt.subplot(132)
sp1.imshow(panc.sliceSag,cmap='gray',clim=(-200,300),aspect=sideratio)
sp1=plt.subplot(133)
sp1.imshow(panc.sliceCor,cmap='gray',clim=(-200,300),aspect=sideratio)
plt.show()