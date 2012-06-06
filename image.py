# -*- coding: utf-8 -*-
# Nifti Image load and save
import numpy as np
import nibabel as nib
import os
import dicom
import shutil
from datetime import datetime
def loadnii(folder,name):
    """ Load Nifti file and parse it to image and header """
    img = nib.load(os.path.join(folder,name))
    data = img.get_data()
    hdr = img.get_header()
    affinemrx = hdr.get_sform()
    return data , hdr, affinemrx
def savenii(data,matrix,folder):
    nib.nifti1.save(nib.Nifti1Image(data, matrix),folder)
def crop(data,xc,yc,zc,size=30,invert=True):
    x,y,z,t=np.shape(data)
    if invert == True:
        xc,yc,zc=np.array([x,y,z])-np.array([xc,yc,zc])
    data = data[xc-size:xc+size,yc-size:yc+size,zc-size:zc+size]# crop data
    return data

def convert(img,folder):
#convert 4d image
    a=nib.funcs.four_to_three(img)
    for i in range(len(a)):
        nib.nifti1.save(a[i],folder+'%s'% i)

def dcm_parser(folder, folder_out=None,force=False,):
    """ parse DICOM in folder and copy it in folder_out
with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/"""
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

    for pathfold,dirs,file_list in os.walk(folder):

        for file_name in file_list:

            try:
                dcm=dicom.read_file( os.path.join(pathfold,file_name),force=force )
                out_path=os.path.join(folder_out,
                                        tname(dcm,"PatientsName"),
                                        tname(dcm,"StudyDate")+'_'+tname(dcm,"StudyID"),
                                        tname(dcm,"SeriesNumber")+'_'+'_'+tnum(dcm,0x7005101b)+'_'+tnum(dcm,0x7005100b))

                shutil.move(os.path.join(pathfold,file_name),out_path+'/')

            except dicom.filereader.InvalidDicomError as (s):
                log.write ("Can't read file %s in %s : "%(file_name,pathfold) + str(s)+'\n')
                continue
            except IOError as (s):
                os.makedirs(s.filename)
                #noinspection PyUnboundLocalVariable
                shutil.move(os.path.join(pathfold,file_name),out_path+'/')
                continue
            except shutil.Error:
                log.write ('Error moving %s to %s'%(file_name,out_path)+'\n')
                continue
    log.close()


def transconvert(mrxfileSlicer='stack.tfm',folder='/media/63A0113C6D30EAE8/_PERF/YAVNIK/slicer/',inputim=''):

    mrxfileMango="Mango"+mrxfileSlicer[:-4]+'.txt'
    mrxfileMangoAbs=folder+mrxfileMango

    os.mknod(mrxfileMangoAbs)
    mrxSlicer=open(folder+mrxfileSlicer)
    mrxSlicer = mrxSlicer.readlines()[3].split()[1:]
    mrxMango=[mrxSlicer[i:i+3] for i in [0,3,6]]
    mrxMango=[mrxMango[i-1]+[mrxSlicer[-i]] for i in [1,2,3]]

    for i in mrxMango:
        open(mrxfileMangoAbs,'a').write(' '.join(i)+'\n')

    os.system('applytransform -c %s %s %s'%(mrxfileMangoAbs,inputim,inputim+'_transformed'))