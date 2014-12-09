import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from scipy import signal
from scipy.ndimage import filters
import os

import pyximport
pyximport.install()
import dist_calc


class roi2():
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
        img_new = nibabel.nifti1.nifti1image(self.d_map, none)
        print self.affinemrx

        nibabel.nifti1.save(img_new, os.path.join(self.path[:-4] + 'new.nii'))


#image_folder = 'd:/nest_arj/tips_generalbodyperfusionyavnikga12081948s005a001_fc17reg_9_2_1000_roi.nii'
#r = roi(image_folder)
#r.distance_map()
#r.save()

def distace_kernel(kernel_size,dim_res):

    voxel_size=np.asarray(dim_res, dtype=float)
    #hack to override 4th axis of nii image
    if len(dim_res)==4:
        voxel_size=voxel_size[:3]


    x,y,z=np.ceil(kernel_size/voxel_size)
    #make 3d grid of euclidean distances from center
    distances=voxel_size*np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    #print distances
    distances=np.sqrt(np.sum( distances*distances ))

    #/ np.sqrt( np.pi*2*sigma**2)**3
    if len(dim_res)==4:
        distances=distances[...,None]
    print distances.shape
    return distances

def get_min_distance(data,d_map):
    return min(d_map[data==0])

def make_map(data,data_res,kernel_size=60):
    if data.ndim>3:
        data_res=data_res[:3]
        data=data[...,0]
    dkernel = distace_kernel(kernel_size,data_res)
    dkernel = np.asarray(dkernel, dtype=np.float32)
    data = np.asarray(data, dtype='int32')

    out = dist_calc.dict_calc(data, dkernel)
    print 'Gotcha!'
    return out

def make_map_nii(input_file_path,kernel_size=60):
    f = nib.load(os.path.abspath(input_file_path))
    vol = f.get_data()
    hdr = f.get_header()
    res = hdr.get_zooms()  # getting voxel size
    mrx = hdr.get_sform()  # getting affine matrix
    print hdr
    print mrx
    print res
    map_image = make_map(vol,res,kernel_size)

    nii_map = nib.Nifti1Image(map_image, mrx, hdr)

    input_fname = os.path.basename(os.path.abspath(input_file_path)).split('.')[0]
    input_path = os.path.split(input_file_path)[0]

    nib.nifti1.save(nii_map,
                    os.path.join(input_path , input_fname + "_dist_map.nii"))

make_map_nii('/media/WORK___/_PERF/LAGUNOVA  V.V. 21.01.1939/20140113_20/ROI/tumor_roi2.nii.gz',kernel_size=20)


