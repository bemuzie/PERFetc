import nibabel as nib
import os
import numpy as np


class Roi():
    def __init__(self, roi_file, rois_info=None):
        # loading rois in folder
        # forming vol name = roi_name - roi*
        f = nib.load(os.path.abspath(roi_file))
        self.roi_vol = f.get_data()
        if not rois_info:
            root = os.path.abspath(os.path.join(rois_folder, '..', 'ROI'))
            rois_info = os.path.join(root, os.path.basename(rois_folder), '_info.txt')
        self.__read_info(rois_info)
        self.__get_concentrations()
        self.mean_concentrations = [np.mean(i) for i in self.concentrations]
        self.max_hu = max(self.mean_concentrations)
        self.max_t = self.mean_concentrations.index(self.max_hu)

    def __read_info(self, rois_info):
        f = open(rois_info, 'r')
        r_folder = f.read_lines()[0]
        self.vol_names = {}

        for i in f.read_lines()[1:]:
            ii = i.split(',')
            self.vol_names[os.path.join(r_folder, ii[0])] = ii[1]

    def __get_concentrations(self):
        self.concentrations = []
        for vol_path, vol_time in self.vol_names.items():
            vol_f = nib.load(os.path.abspath(vol_path))
            vol = vol_f.get_data()
            self.concentrations += vol[self.roi_vol == 1],
            self.times += vol_time
        self.concentrations = [c for (t, c) in sorted(zip(self.times, self.concentrations))]
        self.times = sorted(self.times)

    def output(self):
        print self.mean_concentrations
        print self.max_hu
        print self.max_t


