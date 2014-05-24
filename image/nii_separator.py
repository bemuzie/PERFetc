import nibabel as nib
import os

def separate_nii(input_file, output_folder=None):
    """
    Separate 4d *.nii file to 3d *.nii files and write them in output directory.

    Args:
      input_file (str): absolute path to 4d nii file
      output_folder (str, optional): absolute path where 3d nii files will be saved. 
	      If not given they will be saved in the same folder subfolder named like infut file.

    Returns:
      True

    Raises:
      AttributeError: The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
      ValueError: If `param2` is equal to `param1`.
    """


    if not output_folder:

    	new_folder = os.path.basename(os.path.abspath(input_file)).split('.')[0] #naming output folder like file without extension
    	output_folder = os.path.join(os.path.dirname(os.path.abspath(input_file)), new_folder)


    vol4d = nib.load(os.path.abspath(input_file)) # loading 4d image
    vol3d_list=nib.funcs.four_to_three(vol4d) # getting list of 3d volumes

    for i in range(len(vol3d_list)):
    	try:
        	nib.nifti1.save(vol3d_list[i], 
        					os.path.join(output_folder, '%s_%s.nii'%(new_folder,i)))
        except IOError, s:
        	if s[0] == 2: #No directory exception
        		os.mkdir(os.path.join(output_folder))
        		nib.nifti1.save(vol3d_list[i], 
        					os.path.join(output_folder, '%s_%s.nii'%(new_folder,i)))
        		continue



    return True

def merge_nii(input_folder):
	pass

def get_resolution(input_file):
    vol4d = nib.load(os.path.abspath(input_file))
    hdr = vol4d.get_header()
    print hdr.get_zooms()

separate_nii('/home/denest/PERF_volumes/DASHKOV A.P. 05.09.1939/NII/20140521_102800GeneralBodyPerfusionCopiedDASHKOVAP05091939s008a001.nii')