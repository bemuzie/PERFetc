import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter
import nibabel
import os 

def sphere_mask(radius):
	sl=slice(-radius,radius+1)
	mrx=np.ogrid[sl,sl,sl]
	mrx=[i**2 for i in mrx]
	eucl_mrx = np.sqrt(np.sum(mrx))
	eucl_mrx[eucl_mrx<=radius]=1
	eucl_mrx[eucl_mrx>radius]=0
	print eucl_mrx.shape
	return eucl_mrx

def isLesion(input_data,av_down,av_up,sd_down,sd_up):
	try:

		
		d_av=np.average(input_data)
		

		
		if d_av > av_down and d_av<av_up:
			d_sd=np.std(input_data)
			if d_sd > sd_down and d_sd < sd_up:
				
				return 1.
			else: 
				
				return 0
		else:
			
			return 0

	except:
		
		print input_data
		return 10

IMAGE_FOLDER='/home/denest'
IMAGE_NAME='13 Venous Phase  1.5  B30f-subvolume-scale_1.nii'
img=nibabel.nifti1.load(os.path.join(IMAGE_FOLDER,IMAGE_NAME))
hdr = img.get_header()
affinemrx = hdr.get_sform()
data=img.get_data()
print np.array(data,dtype=float).shape
data_new=generic_filter(np.array(data,dtype=float), isLesion , footprint=sphere_mask(2) ,extra_keywords={
	'av_down':148.,'av_up':152.,'sd_down':15,'sd_up':17.})

img_new=nibabel.nifti1.Nifti1Image(data_new,affinemrx)
nibabel.nifti1.save(img_new, os.path.join(IMAGE_FOLDER,IMAGE_NAME[:-4]+'masked.nii'))
