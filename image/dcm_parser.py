# -*- coding: utf-8 -*-
# Nifti Image load and save
# -*- coding: utf-8 -*-
# Nifti Image load and save
import numpy as np
# import nibabel as nib
import os
import dicom
import shutil
from datetime import datetime
import time


class Loger():
    def __init__(self, adress):
        curtime = datetime.now()
        self.log = open(adress + '\log_' + str(curtime.month) + str(curtime.day) + '_' + str(curtime.hour) + str(
            curtime.minute) + str(curtime.second) + '.txt', 'w')
        self.num_moved = 0
        self.num_exist = 0
        self.num_unread = 0

    def moved(self, prnt=True):
        self.num_moved += 1

    def unread(self, prnt=True):
        self.num_unread += 1

    def exist(self, prnt=True):
        self.num_exist += 1

    def finish(self):
        self.log.write('%s where succesfully moved' % self.num_moved + '\n')
        self.log.write('%s are already exist' % self.num_exist + '\n')
        self.log.write('%s where not read' % self.num_unread + '\n')
        self.log.close()


class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def get_times(folder, folder_out=None, force=False, ):
    """ parse DICOM in folder and copy it in folder_out
    with folder structure /PatientName-BirthDate/StudyNumber/SeriesNumber/"""
    a = 0
    i = 0
    curtime = datetime.now()
    log = open(folder_out + '/log_' + str(curtime.month) + str(curtime.day) + '_' + str(curtime.hour) + str(
        curtime.minute) + str(curtime.second) + '.txt', 'w')

    def tname(img, tagname):
        try:
            sfolder = str(img.data_element(tagname)._value)
        except:
            sfolder = 'None'
            pass
        return sfolder

    def tnum(img, tagnum):
        try:
            sfolder = str(img[tagnum]._value)
        except:
            sfolder = 'None'
            pass
        return sfolder

    def time_str_to_float(time_str):
        t = map(float, [time_str[:2], time_str[2:4], time_str[4:6], time_str[6:]])
        coef = [360, 60, 1, 0.001]
        return int(round(sum([i * ii for i, ii in zip(t, coef)]), 0))

    output_dictionary = {'series_time': []}

    if not folder_out:
        folder_out = folder
    for pathfold, dirs, file_list in os.walk(folder):
        dcm_list = filter(lambda x: '' in x, file_list)
        a += len(dcm_list)
    print a
    moved = 0
    unmoved = 0
    notread = 0
    output_file = open(folder_out + '/time_info.txt', 'w')
    output_dict = Vividict()
    for pathfold, dirs, file_list in os.walk(folder):

        for file_name in file_list:

            try:
                fpath = os.path.join(pathfold, file_name)
                dcm = dicom.read_file(fpath, force=force)

                series_num = tname(dcm, "SeriesNumber")
                series_time = tnum(dcm, 0x00080031)
                acquisition_time = tnum(dcm, 0x00080032)
                acquisition_num = tnum(dcm, 0x00200012)
                #study_time=tnum(dcm,0x00080030)
                contrast_time = tnum(dcm, 0x00181042)
                output_dict[contrast_time][series_num][series_time][acquisition_num]
                output_dict[contrast_time][series_num][series_time][acquisition_num] = acquisition_time


            except dicom.filereader.InvalidDicomError as (s):
                log.write("Can't read file %s in %s : " % (file_name, pathfold) + str(s) + '\n')
                notread += 1
                continue
            except shutil.Error:
                log.write('Error moving %s to %s' % (file_name, out_path) + '\n')
                unmoved += 1
                continue
    print output_dict
    # write csv in format: 'series_num'_'aquisition_num','time','is it 2nd cm injection'

    cm_injections = [time_str_to_float(i) for i in output_dict.keys() if not i == 'None']

    for contrast_time, sn in output_dict.items():
        second_injection = 0
        if not contrast_time == 'None' and len(cm_injections) > 1 and time_str_to_float(contrast_time) == max(
                cm_injections):
            second_injection = 1
        for series_num, st in sn.items():
            for series_time, a_num in st.items():
                print a_num
                for acquisition_num, acquisition_time in a_num.items():
                    if contrast_time == 'None':
                        output_file.write('%s_%s,0,0\n' % (series_num, acquisition_num))
                    else:
                        phase_time = str(time_str_to_float(acquisition_time) - time_str_to_float(contrast_time))
                        output_file.write('%s_%s,%s,0\n' % (series_num, acquisition_num, phase_time))


if __name__ == "__main__":
    Parser('/home/denest/_TEMP', '/home/denest/PERF_volumes')