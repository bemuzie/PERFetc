__author__ = 'ct'
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import radon,iradon
import image
import numpy as np

folder='/media/63A0113C6D30EAE8/_CT/IVANOVA N.A. 03.07.1954/Nifti/'
vol,hdr,mrx=image.loadnii(folder,'20110215_1324203PhaseLiverPORTAEIVANOVANA03071954s014a004.nii')
axial=np.rot90(vol[...,160])
projections=radon(axial,np.linspace(0,360,1000))
print np.shape(projections)

imgplt=plt.subplot(131)
reconplt=plt.subplot(132)
reconplt2=plt.subplot(133)


imgplt.imshow(axial,cmap='gray',clim=[-200,200])
reconplt.imshow(iradon(projections[100:-100],theta=np.linspace(0,360,1000)),cmap='gray',clim=[-200,200])
reconplt2.imshow(iradon(projections,theta=np.linspace(0,360,1000)),cmap='gray',clim=[-200,200])

plt.show()