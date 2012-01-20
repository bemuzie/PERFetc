# Nifti Image load and save
import numpy as np
#import nibabel as nib
from os import *
import timeit
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
    """ parse DICOM in folder and copy it in folder_out
with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/"""
    import dicom
    import shutil
    
    if folder_out == []:
        folder_out=folder
    ls=listdir(folder)
    for i in ls:
        try:
            ds=dicom.read_file(path.join(folder,i))
            out_path=path.join(folder_out,ds.PatientsName,str(ds.StudyDate)+'_'+str(ds.StudyID),str(ds.SeriesNumber))
            shutil.copy(path.join(folder,i),out_path+'/')
        except IOError as (s):
                makedirs(s.filename)
                shutil.copy(path.join(folder,i),out_path+'/')
        continue
    print len(listdir(folder))
    print len(listdir(out_path))
     
    
func=dcm_parser('/media/63A0113C6D30EAE8/_PERF/temp/perf_20111228_111309207/','/media/63A0113C6D30EAE8/_CT/')
t = timeit.Timer(stmt="func", setup="from __main__ import * ")
print t.timeit()
