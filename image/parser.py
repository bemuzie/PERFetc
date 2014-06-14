# -*- coding: utf-8 -*-
import numpy as np
import os
import dicom
import shutil
from datetime import datetime
import gzip


class Parser():
    def __init__(self, folder, folder_out, force=False):
        self.folder = folder
        self.folder_out = folder_out
        self.force = force
        self.logger = Loger(folder_out)

        for pathfold, dirs, file_list in os.walk(folder):
            for file_name in file_list:
                in_file = os.path.join(pathfold, file_name)
                print in_file
                try:
                    out_file_p, out_file_n = self.file_read(in_file)
                    self.file_copy(in_file, out_file_p, out_file_n)
                except dicom.filereader.InvalidDicomError as (s):
                    print "вшсщь уккщк", s
                    self.logger.unread()
                    continue
                except shutil.Error as (s):
                    print 'main loop', s
                    self.logger.exist()
                    continue
        self.logger.finish()


    def tname(self, img, tagname):
        try:
            sfolder = str(img.data_element(tagname)._value).strip(' .')
        except:
            sfolder = 'None'
            pass
        return sfolder

    def tnum(self, img, tagnum):
        try:
            sfolder = str(img[tagnum]._value).strip(' .')
        except:
            sfolder = 'None'
            pass
        return sfolder

    def file_read(self, file_path):
        dcm = dicom.read_file(file_path, force=self.force)
        file_path = os.path.join(self.folder_out,
                                 self.tname(dcm, "PatientsName"),
                                 self.tname(dcm, "StudyDate") + '_' + self.tname(dcm, "StudyID"),
                                 self.tname(dcm, "SeriesNumber") + '_' + self.tnum(dcm, 0x7005101b) + '_' + self.tnum(
                                     dcm, 0x7005100b))
        file_name = self.tnum(dcm, 0x00080018) + '.dcm'
        return file_path, file_name


    def file_copy(self, in_path, out_path, file_name):
        try:
            shutil.copy2(in_path, os.path.join(out_path, file_name))
            self.logger.moved()
        except IOError as s:
            print "move error", s
            os.makedirs(out_path)
            shutil.copy2(in_path, os.path.join(out_path, file_name))
            self.logger.moved()
            pass


class Loger():
    def __init__(self, adress):
        curtime = datetime.now()
        self.log = open(adress + '\log_' + str(curtime.month) + str(curtime.day) + '_' + str(curtime.hour) + str(
            curtime.minute) + str(curtime.second) + '.txt', 'w')
        self.num_moved = 0
        self.num_exist = 0
        self.num_unread = 0

    def moved(self):
        self.num_moved += 1

    def unread(self):
        self.num_unread += 1

    def exist(self):
        self.num_exist += 1

    def finish(self):
        self.log.write('%s where succesfully moved' % self.num_moved + '\n')
        self.log.write('%s are already exist' % self.num_exist + '\n')
        self.log.write('%s where not read' % self.num_unread + '\n')
        self.log.close()
        
