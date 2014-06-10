# -*- coding: utf-8 -*-
# Nifti Image load and save
# -*- coding: utf-8 -*-
# Nifti Image load and save
import numpy as np
#import nibabel as nib
import os
import dicom
import shutil
from datetime import datetime
import time


class Parser():
    def __init__(self,folder,folder_out,folder_structure=None,force=False):
        
        self.folder = folder
        self.folder_out = folder_out
        self.force=force
        self.logger = Loger(folder_out)
        self.todo = 0
        self.done = 0
        #for pathfold,dirs,file_list in os.walk(folder):
        #    self.todo+=len(file_list)
    
        for pathfold,dirs,file_list in os.walk(folder):
            for file_name in file_list:
                in_file = os.path.join(pathfold,file_name)
                self.done+=1
                try:
                    out_file_p,out_file_n = self.file_read(in_file)
                    if out_file_p and out_file_n:
                        self.file_copy(in_file,out_file_p,out_file_n)
                
                except shutil.Error as (s):
                    self.logger.exist()
                    continue
        self.logger.finish()

    
    def tname(self,img,tagname):
        try:
            sfolder=str(img.data_element(tagname)._value).strip(' .')
        except:
            sfolder='None'
            pass
        return sfolder
        
    def tnum(self,img,tagnum):
        try:
            sfolder=str(img[tagnum]._value).strip(' .')
        except:
            sfolder='None'
            pass
        return sfolder
    def file_read(self,file_path):
        #print file_path
       
        try:
            with open(file_path,'r+b') as f:
                dcm=dicom.read_file( f,force=self.force, stop_before_pixels=True)
            
            file_path= os.path.join(self.folder_out, 
                            self.tname(dcm,"PatientsName"),
                            self.tname(dcm,"StudyDate")+'_'+self.tname(dcm,"StudyID"),
                            self.tname(dcm,"SeriesNumber")+'_'+self.tnum(dcm,0x7005101b)+'_'+self.tnum(dcm,0x7005100b))
            file_name= self.tnum(dcm,0x00080018)+'.dcm'
        except dicom.filereader.InvalidDicomError as (s):
            #print s
            #print f.closed
            self.logger.unread()
            return None,None
        except IOError as (s):
            #print s
            time.sleep(15)
            return None,None
            
        return file_path,file_name
    
        
        
    def file_copy(self,in_path,out_path,file_name):
        try:
            shutil.copy2(in_path,os.path.join(out_path,file_name))
            os.remove(in_path)
            self.logger.moved()
        except IOError as s:
            #print "move error",s
            os.makedirs(out_path)
            shutil.copy2(in_path,os.path.join(out_path,file_name))
            os.remove(in_path)
            self.logger.moved()
            pass
        
 
class Loger():
    def __init__(self,adress):
        curtime=datetime.now()
        self.log = open( adress+'\log_'+str(curtime.month)+str(curtime.day)+'_'+str(curtime.hour)+str(curtime.minute)+str(curtime.second)+'.txt' , 'w')
        self.num_moved=0
        self.num_exist=0
        self.num_unread=0
    def moved(self,prnt=True):
        self.num_moved+=1
    def unread(self,prnt=True):
        self.num_unread+=1
    def exist(self,prnt=True):
        self.num_exist+=1
    def finish(self):
        self.log.write('%s where succesfully moved'%self.num_moved + '\n')
        self.log.write('%s are already exist'%self.num_exist + '\n')
        self.log.write('%s where not read'%self.num_unread + '\n')
        self.log.close()
        


if __name__ == "__main__":

    Parser('/home/denest/_TEMP','/home/denest/PERF_volumes')