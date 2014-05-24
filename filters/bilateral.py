import numpy as np
import pyximport; pyximport.install()
import cython_bilateral3d
from scipy import ndimage
import nibabel as nib
import os


def bilateral(input_file,output_folder=None,sig_i=40,sig_g=1,x_range=[20,512-20], y_range=[20,512-20], z_range=[20,300]):
    """
    Process input 3d nii file with bilateral filter with 3d kernel


    Args:
      input_file (str): absolute path to 3d nii file
      output_folder (str, optional): absolute path where filtered nii files will be saved. 
          If not given they will be saved in the same folder subfolder named like infut file.
      sig_i (float): Intensity standard deviation, usual levels are 20-40
      sig_g (float): Distanse standard deviation in mm.  Filter diametr will be 4*sig_g in each axis.
      x_range,y_range,z_range ([int,int]) : croping range of image in each axis.

    Returns:
      True
    
    """
    f = nib.load(os.path.abspath(input_file))
    vol3d = f.get_data()
    hdr = f.get_header()
    res = hdr.get_zooms() # getting voxel size
    mrx = hdr.get_sform() # getting affine matrix

    vol3d = np.array(vol3d, dtype='int32',order='C')
    filtered_image = cython_bilateral3d.bilateral3d(vol3d, res,
                                                              sig_g,
                                                              sig_i,
                                                              x_range=x_range, y_range=y_range, z_range=z_range)

    file_name_base = os.path.basename(os.path.abspath(input_file)).split('.')[0]
    if not output_folder:

      new_folder = os.path.basename(os.path.abspath(input_file)).split('.')[0]+'_filtered' #naming output folder like file without extension
      output_folder = os.path.join(os.path.dirname(os.path.abspath(input_file)), new_folder)

    else:
      new_folder = output_folder

    try:
        nib.nifti1.save(nib.Nifti1Image(filtered_image, mrx), 
                    os.path.join(output_folder, "%s_I%s_G%s.nii"%(file_name_base,sig_i,sig_g)) )

    except IOError, s:
            if s[0] == 2: #No directory exception
                os.mkdir(os.path.join(output_folder))
                nib.nifti1.save(nib.Nifti1Image(filtered_image, mrx), 
                                os.path.join(output_folder, "%s_I%s_G%s.nii"%(file_name_base,sig_i,sig_g)) )

#bilateral('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_8.nii',
#          sig_i=40,
#          sig_g=2)