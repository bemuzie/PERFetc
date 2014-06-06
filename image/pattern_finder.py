import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter
from scipy.ndimage import filters
from scipy.ndimage import fourier
import nibabel
import os 
from skimage import measure
from skimage.filter import median_filter


def sphere_mask(radius,radius2=None):
	mr=max(radius,radius2)
	sl=slice(-mr,mr+1)
	mrx=np.ogrid[sl,sl,sl]
	mrx=[i**2 for i in mrx]
	eucl_mrx = np.sqrt(np.sum(mrx))
	mrx_footprint,mrx_map=np.zeros(eucl_mrx.shape),np.zeros(eucl_mrx.shape)
	mrx_footprint[eucl_mrx<=mr]=1
	mrx_map[eucl_mrx<=mr]=2
	mrx_map[eucl_mrx<=min(radius,radius2)]=1
	
	print eucl_mrx.shape
	return mrx_footprint, mrx_map

def isLesion(input_data,av_down,av_up,sd_down,sd_up):
	try:

		
		d_av=np.average(input_data)
		

		
		if d_av > av_down and d_av<av_up:
			d_sd=np.std(input_data)
			if d_sd < sd_up:
				
				return 1.
			else: 
				
				return 0
		else:
			
			return 0

	except:
		
		print input_data
		return 10
def isHypoLesion(input_data,diff,data_map,sd=30):
	lesion=input_data[data_map==1]
	lesion_density=np.average(lesion)
	background_density=np.average(input_data[data_map==2])
	if background_density-lesion_density > diff:
		if np.std(lesion)<sd:
			return 1
		else:
			return 0
	else:
		return 0
def AVvsBackground(data,lesion_size,penumbra_size):
	penumbra_mask=sphere_mask(lesion_size,penumbra_size)[1]
	penumbra_mask[penumbra_mask<2]=0
	penumbra_mask[penumbra_mask==2]=1
	average_lesion=generic_filter(data,np.average,footprint=sphere_mask(lesion_size)[0])
	average_penumbra=generic_filter(data,np.average,footprint=penumbra_mask)

	return average_penumbra-average_lesion

def SDvsBackground(data,lesion_size,penumbra_size):
	penumbra_mask=sphere_mask(lesion_size,penumbra_size)[1]
	penumbra_mask[penumbra_mask<2]=0
	penumbra_mask[penumbra_mask==2]=1
	average_lesion=generic_filter(data,np.std,footprint=sphere_mask(lesion_size)[0])
	average_penumbra=generic_filter(data,np.std,footprint=penumbra_mask)

	return average_lesion



class Canny_detector():
	def __init__(self,data,kernel):
		self.data=data
		self.kernel=kernel
		maximum=1000
		minimum=0
	def reduce_nonmax(self):
		maximums=filters.maximum_filter(self.data,self.kernel)




def canny(data):
	center=data[data.size/2]

	if center == np.max(data):
		return center
	else:
		return center/2.

IMAGE_FOLDER='D:\DICOM'
IMAGE_NAME='IM00407-subvolume-scale_1.nii'
img=nibabel.nifti1.load(os.path.join(IMAGE_FOLDER,IMAGE_NAME))
hdr = img.get_header()
affinemrx = hdr.get_sform()
data=img.get_data()

dmap=np.ravel(sphere_mask(3,2)[1])
dmap=dmap[dmap>0]
"""
data_new=generic_filter(np.array(data,dtype=float), 
						isHypoLesion ,
						footprint=sphere_mask(7,2)[0],
						extra_keywords={
										'diff':10.,
										'data_map':dmap,
										'sd':30.,
										})
"""

data_new=filters.gaussian_gradient_magnitude(filters.median_filter(data,3), 1.5)

data_new=filters.generic_filter(data_new,canny,3)
data_new=filters.generic_filter(data_new,canny,3)



img_new=nibabel.nifti1.Nifti1Image(data_new,affinemrx)
nibabel.nifti1.save(img_new, os.path.join(IMAGE_FOLDER,IMAGE_NAME[:-4]+'gaussian_laplace2.nii'))


