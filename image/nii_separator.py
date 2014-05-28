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
    v4=vol4d.get_data()
    hdr=vol4d.get_header()
    print hdr

    vol3d_list=nib.funcs.four_to_three(vol4d) # getting list of 3d volumes

    for i in range(len(vol3d_list)):
    	try:
        	nib.nifti1.save(vol3d_list[i], 
        					os.path.join(output_folder, '%s_%s.nii.gz'%(new_folder,i)))
        except IOError, s:
        	if s[0] == 2: #No directory exception
        		os.mkdir(os.path.join(output_folder))
        		nib.nifti1.save(vol3d_list[i], 
        					os.path.join(output_folder, '%s_%s.nii.gz'%(new_folder,i)))
        		continue



    return True

def merge_nii(input_folder):
	pass

def get_resolution(input_file):
    vol4d = nib.load(os.path.abspath(input_file))
    hdr = vol4d.get_header()
    print hdr.get_zooms()

def crop_image(input_file,x1,x2,y1,y2,z1,z2,output_file =None):
    img = nib.load(os.path.abspath(input_file))

    f = nib.load(os.path.abspath(input_file))
    vol = f.get_data()
    hdr = f.get_header()
    res = hdr.get_zooms() # getting voxel size
    mrx = hdr.get_sform() # getting affine matrix
    print hdr
    print mrx
    print res
    filtered_image=vol[-x2:-x1,-y2:-y1,-z2:-z1]

    if output_file ==None:
        output_file = os.path.join(os.path.dirname(os.path.abspath(input_file)))
    
    input_fname = os.path.basename(os.path.abspath(input_file)).split('.')[0]
        
    nii_im = nib.Nifti1Image(filtered_image, mrx,hdr)

    nib.nifti1.save(nii_im, 
                        os.path.join(output_file, input_fname + "X%s_%s_Y%s_%s_Z%s_%s.nii"%(x1,x2,y1,y2,z1,z2)) )

def set_header_from(image,image_header):
    f1 = nib.load(os.path.abspath(image_header))
    hdr = f1.get_header()
    res = hdr.get_zooms() # getting voxel size
    mrx = hdr.get_sform() # getting affine matrix

    changing_image = nib.load(os.path.abspath(image))
    d=changing_image.get_data()
    
    nib.nifti1.save(nib.Nifti1Image(d,mrx,hdr), image)

if __name__ == '__main__':
    set_header_from('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/pancreas_roi.nii.gz','/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_6_I40_G1.5.nii')
    crop_image('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/pancreas_roi.nii.gz',
                84,329,160,290,1,260)