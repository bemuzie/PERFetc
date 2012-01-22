# Nifti Image load and save
import numpy as np
import nibabel as nib
import os
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
        nib.nifti1.save(a[i],folder+'%s'% i)

def dcm_parser(folder, folder_out=None):
    """ parse DICOM in folder and copy it in folder_out
with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/"""
    global out_path
    import dicom
    import shutil
    
    if not folder_out:
        folder_out=folder
    ls=os.listdir(folder)
    for i in ls:
        try:
            ds=dicom.read_file(os.path.join(folder,i))
            out_path=os.path.join(folder_out,ds.PatientsName,str(ds.StudyDate)+'_'+str(ds.StudyID),str(ds.SeriesNumber))
            shutil.copy(os.path.join(folder,i),out_path+'/')
        except IOError as (s):
                os.makedirs(s.filename)
                #noinspection PyUnboundLocalVariable
                shutil.copy(os.path.join(folder,i),out_path+'/')
        continue
    print len(os.listdir(folder))
    #noinspection PyUnboundLocalVariable
    print len(os.listdir(out_path))

