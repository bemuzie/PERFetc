# Nifti Image load and save
import numpy as np
import nibabel as nib
from os import *

def loadnii(folder,name):
    """ Load Nifti file and parse it to image and header """
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

def dcm_parser(folder,folder_out=[]):
    if folder_out == []:
        folder_out=folder
    import dicom
    ls=listdir(folder)
    for i in ls:
        ds=dicom.read_file(path.join(folder,i))
        if ds.PatientsName in listdir(folder_out):
            if ds.SeriesNumber in listdir(path.join(folder_out,ds.PatientsName)):
                
    
    
