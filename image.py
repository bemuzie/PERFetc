# -*- coding: utf-8 -*-
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
    print data.ndim
    nib.nifti1.save(nib.Nifti1Image(data, np.eye(4),hdr),folder)
def crop(data,xc,yc,zc,size=30):
    data = data[xc-size:xc+size,yc-size:yc+size,zc-size:zc+size]# crop data
    return data

def convert(img,folder):
#convert 4d image
    a=nib.funcs.four_to_three(img)
    for i in range(len(a)):
        nib.nifti1.save(a[i],folder+'%s'% i)

def dcm_parser(folder, folder_out=None):
    """ parse DICOM in folder and copy it in folder_out
with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/"""
    import dicom
    import shutil
    a=0
    i=0
    if not folder_out:
        folder_out=folder
    for pathfold,dirs,file_list in os.walk(folder):
        dcm_list=filter(lambda x: '' in x,file_list)
        a+=len(dcm_list)
    print a
    for pathfold,dirs,file_list in os.walk(folder):
        dcm_list=filter(lambda x: '' in x,file_list)
        for file_name in dcm_list:
            try:
                dcm=dicom.read_file( os.path.join(pathfold,file_name) )
                out_path=os.path.join(folder_out,dcm.PatientsName,str(dcm.StudyDate)+'_'+str(dcm.StudyID),str(dcm.SeriesNumber))
                shutil.copy(os.path.join(pathfold,file_name),out_path+'/')
                i+=1
                os.system('clear')
                print 'скопировано ',i,'из',a
            except IOError as (s):
                os.makedirs(s.filename)
                #noinspection PyUnboundLocalVariable
                shutil.copy(os.path.join(pathfold,file_name),out_path+'/')
                i+=1
                os.system('clear')
                print 'скопировано ',i,'из',a
                continue
            #except InvalidDicomError:continue


