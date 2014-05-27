#-*- coding: utf-8-*-

#The workflow for recieved DICOM in some TEMP folder

#Parse DICOM and move them to DATA_STORAGE folder with generated subfolder structure /Patient_name/DCM/Examination_date/Series_Kernel_Filter
#make information file /Patient_name/DCM/Examination_date/info.txt


#Convert DICOMs to NIFTI, write NIFTI to /Patient_name/NII/(Examination_date)_(Series).nii.gz

#compress DICOMs

#Separate 4d NIIs to 3d NIIs and move them to /Patient_name/NII/(Examination_date)_(Series)/(Examination_date)_(Series)_time.nii.gz

#Filter 3dNIIs with 3d bilateral filter with
#move them to /Patient_name/NII/(Examination_date)_(Series)_filter_I(IntensitySigma)_G(GaussianSigma)/(Examination_date)_(Series)_time_filter_I(IntensitySigma)_G(GaussianSigma).nii.gz
#make / 

#Manual manipulations
#Create ROIs for aorta,IVC
#Choose target phase and make registration
#Create ROIs for pancreas,tumor,tumor1
ANTs_PATH = '/home/denest/ANTs-1.9.v4-Linux/bin/'
TARGET_PHASE =8
MASK = ''
#registration
import subprocess
working_folder = '/home/denest/temnosagatij/croped/'
fixed_im='20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_8_I40_G1.5-subvolume-scale_1.nii.gz'
moved_im='20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_20_I40_G1.5-subvolume-scale_1.nii.gz'
prefix='trans20to8'
registration_parametrs=['-d', '3',
						'--transform', 'Affine[0.75]',
						'--metric',  'MI[%s,%s,0.5,32]'%(fixed_im,moved_im),
						'--convergence', '[100x100x100x100,1e-6,5]',
						'--shrink-factors', '8x5x3x1',
						'--smoothing-sigmas', '6x2x1x1vox',
						'--use-estimate-learning-rate-once',
						'-x %s'%MASK,

						'--transform', 'SyN[0.75]',
						'--metric',  'MI[%s,%s,0.5,32]'%(fixed_im,moved_im),
						'--convergence', '[100x100x10,1e-6,5]',
						'--shrink-factors', '6x4x1',
						'--smoothing-sigmas', '8x5x1vox',
						'--use-estimate-learning-rate-once'
						'-x %s'%MASK,
						'-o',prefix
						]

subprocess.check_call('NSLOTS=4',shell=True)
subprocess.check_call(' '.join([ANTs_PATH+'antsRegistration', ' '.join(registration_parametrs)]),shell=True,cwd=working_folder)
subprocess.check_call(' '.join([ANTs_PATH+'antsApplyTransforms',
						   '-d 3',
						   '-r',fixed_im,
						   '-i',moved_im,
						   '-t [%s0GenericAffine.mat,1]'%prefix,
						   '-t [%s1Warp.nii.gz,0]'%prefix,
						   '-o %s_registered.nii.gz'%prefix])
				,shell=True,cwd=working_folder)
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