__author__ = 'ct'
import image
import os
import nibabel as nib
from scipy import ndimage as ndimage
import numpy as np

adress='/media/WORK/_PERF/SZHANIKOV  O.M. 19.01.1947/Nifti4d/'
volume='GeneralBodyPerfusionSZHANIKOVOM19011947s007a001_FC17QDS.nii'

if volume == None:
    imagelist=os.listdir(adress)
    volume=imagelist[0]
img,hdr,mrx=img,hdr,mrx=image.loadnii(adress,volume)

crp_x=slice(167,378)
crp_y=slice(271 ,371)
crp_z=slice(53,258)

img_crp=img[crp_x,crp_y,crp_z]
print mrx
kern=np.ones((5,5,5,1))
#filtration
img_median=ndimage.generic_filter(img_crp,np.median,footprint=kern)
#roi selecting

# make rois with x,y,z,size
rois=dict(
    roiArt=(0,0,0,0),
    roiPHead=(0,0,0,0),
    roiPTail=(0,0,0,0),
)

class Roi:
    def __init__(self,data,center,radius):
        self.center=dict(x=center(0),y=center(1),z=center(2))
        self.radius=radius
        for i in center: roi1coord[i]=slice(roi1coord[i]-radius,roi1coord[i]+radius)
        self.roicoord=roi1coord
    centx,centy,centz=center

data=img
roi1coord=dict(x=11,y=11,z=11)
coordinates=dict(x=0,y=0,z=0)

roi1_rad=10
for i in roi1coord: roi1coord[i]=slice(roi1coord[i]-roi1_rad,roi1coord[i]-roi1_rad)
roi1_data=data[roi1coord['x'],roi1coord['y'],roi1coord['z']]

for i in coordinates: coordinates[i]=np.arange(-roi1_rad,1+roi1_rad)


x,y,z,t=np.ogrid()
mask=x**2+y**2+z**2



print 'begin saving'
image.convert(nib.Nifti1Image(img_median,mrx),'/media/WORK/_PERF/SZHANIKOV  O.M. 19.01.1947/Nifti4d/series/')
image.savenii(img_median,mrx,'/media/WORK/_PERF/SZHANIKOV  O.M. 19.01.1947/Nifti4d/new4d.nii')
