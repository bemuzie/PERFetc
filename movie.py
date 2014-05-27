
import matplotlib.pyplot as plt
from matplotlib import animation,colors
import nibabel as nib
import os
import numpy as np
import matplotlib.cm as cm


INPUT_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ANTS/movie'
slide_cords = [slice(0,-1),slice(61,62),slice(0,-1)]
roi = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/pancreas_roiX84_329_Y160_290_Z1_260.nii'
reference = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973//NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/croped/short/8.nii'
filenames = sorted([filelist for a1,a2,filelist in os.walk(INPUT_FOLDER)][0])
print filenames
roi_slice = nib.load(roi).get_data()[:,:,70,0]
roi_cm = colors.ListedColormap(['k', 'red'])
roi_cm.set_under('k',alpha=0)


reference_img = nib.load(reference).get_data()[:,:,70]

def animate(nframe):
    path_to_file = os.path.join(INPUT_FOLDER,filenames[nframe])
    pic = nib.load(path_to_file).get_data()[:,:,70]
    print roi_slice.shape
    #plt.cla()
    plt.imshow(pic-reference_img,cmap='gray',vmin=-200, vmax=250)
    plt.contour(roi_slice,1, colors='red')
    #plt.imshow(roi_slice, cmap=roi_cm, interpolation='none', alpha=1,clim=[0.9,1.1])
    plt.savefig('./movie/%s.jpg'%nframe)

for i in range(len(filenames)):
    animate(i)
    
    



