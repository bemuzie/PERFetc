# -*- coding: utf-8 -*-
# Nifti Image load and save
import numpy as np
import nibabel as nib
import os
import dicom
import shutil
import timeit
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

def dcm_parser(folder, folder_out=None,subfolders=None):
    """ parse DICOM in folder and copy it in folder_out
with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/"""
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

                ''' will optimize it or rewrite with gdcm as soon as possible'''
                sfolder=dict.copy(subfolders)
                for key,val in subfolders.iteritems():
                    if val in dcm:
                        sfolder[key]=str(dcm[val].value)
                    else:
                        sfolder[key]=('None')

                sfolder1=sfolder['PatientsName']
                sfolder2=sfolder['StudyDate']+'_'+sfolder['StudyID']
                sfolder3=sfolder['SeriesNumber']+'_'+sfolder['ConvolutionKernel']+'_'+sfolder['FilterType']

                out_path=os.path.join(folder_out,sfolder1,sfolder2,sfolder3)
                shutil.move(os.path.join(pathfold,file_name),out_path+'/')
                i+=1
                #os.system('clear')
                #print 'скопировано ',i,'из',a
            except IOError as (s):
                os.makedirs(s.filename)
                #noinspection PyUnboundLocalVariable
                shutil.move(os.path.join(pathfold,file_name),out_path+'/')
                i+=1
                #os.system('clear')
                #print 'скопировано ',i,'из',a
                continue
            except dicom.filereader.InvalidDicomError:
                continue
            except shutil.Error:
                print 'error moving to %s'%(out_path)
                continue
            #except InvalidDicomError:continue


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