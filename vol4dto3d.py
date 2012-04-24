__author__ = 'denis'
import nibabel as nib
import image

vol='/media/WORK/_PERF/KRAMNIK D.D. 02.01.1937/Nifti/GeneralBodyPerfusionKRAMNIKDD02011997s005a003.nii'
folder_out='/media/WORK/_PERF/KRAMNIK D.D. 02.01.1937/Nifti/parsed5/'
image.convert(nib.load(vol),folder_out)