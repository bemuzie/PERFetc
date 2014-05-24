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

#calculte rois parametrs
from perfusion import roi
aorta = roi.Roi('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_3_I40_G1_roi_aorta.nii.gz',
			rois_info='/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/aorta_info.csv')
aorta.output()
ivc = roi.Roi('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_3_I40_G1_roi_ivc.nii.gz',
			rois_info='/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/aorta_info.csv')
ivc.output()
print aorta.equilibrium_time(aorta.concentrations,ivc.concentrations)
pancreas = roi.Roi('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/pancreas_roi.nii.gz',
			rois_info='/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/aorta_info.csv')
pancreas.output()


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