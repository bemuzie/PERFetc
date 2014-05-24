# -*- coding: utf-8 -*-
# Nifti Image load and save
import numpy as np
import nibabel as nib
import os
import dicom
import shutil
from datetime import datetime

def dcm_parser(folder, folder_out=None,force=False):
    """ parse *.dcm in folder and move them to the folder_out
    with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/
 

    Args:
      folder (str): absolute path to folder with *.dcm
      folder_out (str, optional): absolute path where 3d nii files will be saved. 
          If not given they will be saved in the same folder subfolder named like infut file.
    forse (bool, optional): default=False, if True dont checl file extension when read it. Need to be true to read DICOM 3.0 format

    Returns:
      True

    Raises:
      
    
    """

    a=0
    i=0
    curtime=datetime.now()
    log = open( folder_out+'/log_'+str(curtime.month)+str(curtime.day)+'_'+str(curtime.hour)+str(curtime.minute)+str(curtime.second)+'.txt' , 'w')
    def tname(img,tagname):
        try:
            sfolder=str(img.data_element(tagname)._value)
        except:
            sfolder='None'
            pass
        return sfolder

    def tnum(img,tagnum):
        try:
            sfolder=str(img[tagnum]._value)
        except:
            sfolder='None'
            pass
        return sfolder

    if not folder_out:
        folder_out=folder
    for pathfold,dirs,file_list in os.walk(folder):
        dcm_list=filter(lambda x: '' in x,file_list)
        a+=len(dcm_list)
    print a
    moved=0
    unmoved=0
    notread=0
    for pathfold,dirs,file_list in os.walk(folder):

        for file_name in file_list:

            try:
                fpath=os.path.join(pathfold,file_name)
                dcm=dicom.read_file( fpath,force=force )
                out_path=os.path.join(folder_out,
                                        tname(dcm,"PatientsName"),
                                        tname(dcm,"StudyDate")+'_'+tname(dcm,"StudyID"),
                                        tname(dcm,"SeriesNumber")+'_'+tnum(dcm,0x7005101b)+'_'+tnum(dcm,0x7005100b))

                shutil.move(fpath,out_path+'/')
                moved+=1
            except dicom.filereader.InvalidDicomError as (s):
                log.write ("Can't read file %s in %s : "%(file_name,pathfold) + str(s)+'\n')
                notread+=1
                continue
            except IOError as (s):
                os.makedirs(s.filename)
                shutil.move(fpath,out_path+'/')
                moved+=1
                continue
            except shutil.Error:
                log.write ('Error moving %s to %s'%(file_name,out_path)+'\n')
                unmoved+=1
                continue
    log.write('%s where succesfully moved'%moved + '\n')
    log.write('%s are already exist'%unmoved + '\n')
    log.write('%s where not read'%notread + '\n')
    log.close()

#TODO separate parser from IO operations
#TODO add csv writer


dcm_parser('/home/denest/_TEMP','/home/denest/PERF_volumes')