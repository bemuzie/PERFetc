# Nifti Image load and save
import numpy as np
import nibabel as nib
import os

def loadnii(folder,name):
    
    img = nib.load(os.path.join(folder,name))
    data = img.get_data()
    hdr = img.get_header()
	return hdr, data
    
def savenii(data,hdr,folder):
    nib.nifti1.save(nib.Nifti1Image(data, np.eye(4),hdr),folder)

def convert(img,folder):
#convert 4d image
    a=nib.funcs.four_to_three(img)
    for i in range(len(a)):
    nib.nifti1.save(a[i],folder+'%s'%(i))
