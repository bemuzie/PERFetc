__author__ = 'denis'
import scipy.ndimage as ndimage
import nibabel as nib
import numpy as np
import image

folder='/media/WORK/_PERF/TIKHEEV YU.V. 19.02.1935/Nifti4d/'
vol='AbdomenHCTNativeTIKHEEVYUV19021935s003a003.nii'

img,hdr,mrx=image.loadnii(folder,vol)
img=ndimage.generic_gradient_magnitude(img,ndimage.generic_filter(img,np.median,3))

image.savenii(img,mrx,folder+'edges0.nii')

