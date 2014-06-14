import nibabel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from scipy import signal
from scipy.ndimage import filters
import os


class Roi():
    def __init__(self, path):
        img = nibabel.nifti1.load(os.path.join(path))
        self.hdr = img.get_header()
        self.affinemrx = self.hdr.get_sform()
        self.data = img.get_data()
        self.path = path

    def distance_map(self):
        self.d_map = np.zeros(self.data.shape)
        self.matrix = np.copy(self.data)

        self.d_map += self.matrix
        for i in range(50):
            self.matrix = filters.generic_filter(self.matrix, self.erode, size=(3, 3, 3, 1))
            self.d_map += self.matrix

            if np.sum(self.matrix) == 0:
                break

            print i

    def erode(self, data):

        d = data[13]

        if d == 1 and 0 in data:
            return 0
        else:
            return d

    def save(self):
        print 1
        img_new = nibabel.nifti1.Nifti1Image(self.d_map, None)
        print self.affinemrx

        nibabel.nifti1.save(img_new, os.path.join(self.path[:-4] + 'new.nii'))


IMAGE_FOLDER = 'D:/nest_arj/TIPS_GeneralBodyPerfusionYAVNIKGA12081948s005a001_FC17reg_9_2_1000_roi.nii'
r = Roi(IMAGE_FOLDER)
r.distance_map()
r.save()