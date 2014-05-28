#-*- coding: utf-8-*-
import os
import subprocess
from image import nii_separator
import nibabel as nib
import numpy as np
#The workflow for recieved DICOM in some TEMP folder

#Parse DICOM and move them to DATA_STORAGE folder with generated subfolder structure /Patient_name/DCM/Examination_date/Series_Kernel_Filter
DCM_FOLDER = 'DCM/20140508_536/'
#make information file /Patient_name/DCM/Examination_date/info.txt



#Convert DICOMs to NIFTI, write NIFTI to /Patient_name/NII/(Examination_date)_(Series).nii.gz
"""
dcm2nii HELP
-4 Create 4D volumes, else DTI/fMRI saved as many 3D volumes: Y,N = Y
-a Anonymize [remove identifying information]: Y,N = Y
-b load settings from specified inifile, e.g. '-b C:\sett1.ini'  
-c Collapse input folders: Y,N = Y
-d Date in filename [filename.dcm -> 20061230122032.nii]: Y,N = Y
-e events (series/acq) in filename [filename.dcm -> s002a003.nii]: Y,N = Y
-f Source filename [e.g. filename.par -> filename.nii]: Y,N = N
-g gzip output, filename.nii.gz [ignored if '-n n']: Y,N = Y
-i ID  in filename [filename.dcm -> johndoe.nii]: Y,N = N
-m manually prompt user to specify output format [NIfTI input only]: Y,N = Y
-n output .nii file [if no, create .hdr/.img pair]: Y,N = Y
-o Output Directory, e.g. 'C:\TEMP' (if unspecified, source directory is used)
-p Protocol in filename [filename.dcm -> TFE_T1.nii]: Y,N = Y
-r Reorient image to nearest orthogonal: Y,N 
-s SPM2/Analyze not SPM5/NIfTI [ignored if '-n y']: Y,N = N
-v Convert every image in the directory: Y,N = Y
-x Reorient and crop 3D NIfTI images: Y,N = N
"""
PATIENT_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/'

DCM2NII_PATH = '/home/denest/mricron/dcm2nii'
ini_path = 'dcm2nii.ini'
output_folder = 'NII/'
dcm2nii_pars =['-4 n',#Create 4D volumes, else DTI/fMRI saved as many 3D volumes: Y,N = Y
			 '-a n',# Anonymize [remove identifying information]: Y,N = Y
			 '-b dcm2nii.ini',#load settings from specified inifile, e.g. '-b C:\set\t1.ini'  
			 #'-c y',#Collapse input folders: Y,N = Y
			 '-d n',#Date in filename [filename.dcm -> 20061230122032.nii]: Y,N = Y
			 '-e y',# events (series/acq) in filename [filename.dcm -> s002a003.nii]: Y,N = Y
			 '-f y',# Source filename [e.g. filename.par -> filename.nii]: Y,N = N
			 '-g y',#gzip output, filename.nii.gz [ignored if '-n n']: Y,N = Y
			 #'-i y',#ID  in filename [filename.dcm -> johndoe.nii]: Y,N = N
			 #'-m n',#manually prompt user to specify output format [NIfTI input only]: Y,N = Y
			 '-n y',#output .nii file [if no, create .hdr/.img pair]: Y,N = Y
			 "-o '%s'"%os.path.join(PATIENT_FOLDER,output_folder),#Output Directory, e.g. 'C:\TEMP' (if unspecified, source directory is used)
			 '-p n',#Protocol in filename [filename.dcm -> TFE_T1.nii]: Y,N = Y
			 '-r n',# Reorient image to nearest orthogonal: Y,N 
			 '-s n',#SPM2/Analyze not SPM5/NIfTI [ignored if '-n y']: Y,N = N
			 '-v y',#Convert every image in the directory: Y,N = Y
			 '-x n',#Reorient and crop 3D NIfTI images: Y,N = N
			 "'%s'"%os.path.join(PATIENT_FOLDER,DCM_FOLDER) #INPUT FOLDER
			]
"""
subprocess.check_call(' '.join([DCM2NII_PATH,' '.join(dcm2nii_pars)] ),
						shell=True)
"""
#compress DICOMs

#Separate 4d NIIs to 3d NIIs and move them to /Patient_name/NII/(Examination_date)_(Series)/(Examination_date)_(Series)_time.nii.gz
"""
for p,d,f in os.walk(os.path.join(PATIENT_FOLDER, 'NII/')):
	
	for fname in f:
		try:
			nii_separator.separate_nii(os.path.join(p,'20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s010a001.nii'))
		except ValueError, s:
			if s == 'Expecting four dimensions':
				continue
"""
#Select crop volume for NIFTies
CROP_VOLUME = 'crop_volume.nii.gz'
cr_vol=nib.load(os.path.join(PATIENT_FOLDER,'ROI',CROP_VOLUME)).get_data()
if len(cr_vol.shape)==4:
	cr_vol = cr_vol[...,0]

borders_vol = np.where(cr_vol==1)
x_fr,y_fr,z_fr = map(np.min,borders_vol)
x_to,y_to,z_to = map(np.max,borders_vol)

f = nib.load(os.path.join(PATIENT_FOLDER,'NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s010a001','20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s010a001_4.nii.gz'))
vol3d = f.get_data()
hdr = f.get_header()
res = hdr.get_zooms() # getting voxel size
mrx = hdr.get_sform() # getting affine matrix
vol3d = vol3d[x_fr:x_to,y_fr:y_to,z_fr:z_to]
nib.nifti1.save(nib.Nifti1Image(vol3d, mrx), os.path.join(PATIENT_FOLDER,'ROI','test_crop.nii') )

#Filter 3dNIIs with 3d bilateral filter with
#move them to /Patient_name/NII/(Examination_date)_(Series)_filter_I(IntensitySigma)_G(GaussianSigma)/(Examination_date)_(Series)_time_filter_I(IntensitySigma)_G(GaussianSigma).nii.gz
#make / 

#Manual manipulations
#Create ROIs for aorta,IVC
#Choose target phase and make registration
#Create ROIs for pancreas,tumor,tumor1
"""
ANTs_PATH = '/home/denest/ANTs-1.9.x-Linux/bin/'
TARGET_PHASE =8
MASK = ''
#registration

WORKING_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/'
images_folder = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/croped/short/'
fixed_im='/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/croped/short/8.nii'
#moved_im='20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_20_I40_G1.5-subvolume-scale_1.nii.gz'
mask = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973//ROI/8_roi.nii.gz'
output_folder = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/registered/'

def registration(moved_image,fixed_image,mask,output_folder):
	
	prefix='%s_to_%s'%(os.path.basename(moved_image).split('.')[0],os.path.basename(fixed_image).split('.')[0])
	print prefix
	registration_parametrs=['-d', '3',
							'--transform', 'Affine[0.75]',
							'--metric',  'MI[%s,%s,0.5,32]'%(fixed_image,moved_image),
							'--convergence', '[100x100x100,1e-6,5]',
							'--shrink-factors', '8x2x1',
							'--smoothing-sigmas', '6x2x2vox',
							'--use-estimate-learning-rate-once',
							'-x %s'%mask,

							'--transform', 'SyN[0.75]',
							'--metric',  'MI[%s,%s,0.5,32]'%(fixed_image,moved_image),
							'--convergence', '[100x100x100,1e-6,5]',
							'--shrink-factors', '6x2x1',
							'--smoothing-sigmas', '8x2x1vox',
							'--use-estimate-learning-rate-once'
							'-x %s'%mask,
							'-o',prefix
							]
	#print ' '.join(registration_parametrs)
	subprocess.check_call('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=3',shell=True)
	subprocess.check_call(' '.join([ANTs_PATH+'antsRegistration', ' '.join(registration_parametrs)]),shell=True,cwd=output_folder)
	subprocess.check_call(' '.join([ANTs_PATH+'antsApplyTransforms',
							   '-d 3',
							   '-r',fixed_image,
							   '-i',moved_image,
							   '-t [%s0GenericAffine.mat,1]'%prefix,
							   '-t [%s1Warp.nii.gz,0]'%prefix,
							   '-o %s_registered.nii.gz'%prefix])
					,shell=True,cwd=output_folder)

for file_name in [f for p,d,f in os.walk(images_folder)][0]:
	fname = os.path.join(images_folder,file_name)
	if not fname==os.path.join(images_folder ,fixed_im):
		registration(os.path.relpath(fname,output_folder),os.path.relpath(fixed_im,output_folder),os.path.relpath(mask,output_folder),output_folder)

"""

#calculte rois parametrs


"""
Subfolder structure
Patient_name/
	|
	|---- DCM/
	|		|----Examination_date/
	|		|		|----Series_Kernel_Filter/
	|		|
	|		|----info.txt
	|
	|---- NII/
	|		|----ExaminationDate_Series_3d/
	|		|----ExaminationDate_Series_3d_filtered_I(IntensitySigma)_G(GaussianSigma)/
	|		|----PatientName_ExaminationDate_Series.nii.gz
	|
	|---- ROI/
	|		|----ExaminationDate_Series_3d/
	|		|----ExaminationDate_Series_3d_filtered_I(IntensitySigma)_G(GaussianSigma)/
	|		|----PatientName_ExaminationDate_Series.nii.gz
	|		|----roi.info
	|
	|----docs/
	|
	|----pics/
	|
	|----reports/
			|----Examination_date_id_report.txt
"""

#Structure
"""
Examination_date_id_report.txt 

ROI
aorta
	contrast apearance time - Время серии, когда в ЗИ-аорта различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
	maximum intensity HU - Максимальная денситометрическая плотность в ЗИ-аорта
	maximum intensity time - Время серии, когда в ЗИ-аорта денситометрическая плотность станет максимальной
	equilibrium concentration HU - Денситометрическая плотность, при которой в ЗИ-аорта различия с ЗИ-нпв(ниже уровня почечных артерий) станут статистически не значимы (Тест Стьюдента, p<0,05)
	equilibrium concentration time - Время серии, в которой различия между ЗИ-аорта и ЗИ-нпв(ниже уровня почечных артерий) станут статистически не значимы (Тест Стьюдента, p<0,05)
pancreas
	volume - объём ЗИ(поджелудочная железа)
	HU - Денситометрическая плотность ЗИ(поджелудочная железа)
	contrast apearance time - Время серии, когда в ЗИ(поджелудочная железа) различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
	maximum intensity HU - Максимальная денситометрическая плотность в ЗИ(поджелудочная железа)
	maximum intensity time - Время серии, когда в ЗИ(поджелудочная железа) денситометрическая плотность станет максимальной
	equilibrium concentration HU - Денситометрическая плотность ЗИ(поджелудочная железа) в equilibrium concentration time
tumor
	volume - объём ЗИ(опухоль)
	max size - максимальный размер зоны интереса опухоль
	HU - Денситометрическая плотность ЗИ(опухоль)

	contrast apearance time - Время серии, когда в ЗИ(опухоль) различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
	maximum intensity HU - Максимальная денситометрическая плотность в ЗИ(опухоль)
	maximum intensity time - Время серии, когда в ЗИ(опухоль) денситометрическая плотность станет максимальной
	maximum intensity diff HU - Максимальная разница между денситометрической плотностью в ЗИ(опухоль) и ЗИ(поджелудочная железа)
	maximum intensity diff time - Время серии максимальной разницы между денситометрической плотностью в ЗИ(опухоль) и ЗИ(поджелудочная железа)
	equilibrium concentration HU - Денситометрическая плотность ЗИ(опухоль) в equilibrium concentration time



"""

#roi.info
"""
roi_name;vol_path,vol_time....
"""