from filters import bilateral
from image.nii_separator import *
import os

INPUT_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/'
OUTPUT_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/croped/'
for pathfold,dirs,file_list in os.walk(INPUT_FOLDER):
	for file_name in file_list:
		if not os.path.exists(os.path.join(OUTPUT_FOLDER,'_'.join([file_name.split('.')[0],'I40','G1.5.nii' ]))):
			print os.path.join(OUTPUT_FOLDER,'_'.join([file_name.split('.')[0],'I40','G1.5.nii' ]))
			fpath=os.path.join(pathfold,file_name)
			crop_image(fpath,84,329,160,290,1,260,'/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/croped/')
			#bilateral.bilateral(fpath,OUTPUT_FOLDER, sig_i=40, sig_g=1.5)
		else:
			print 'passed'


