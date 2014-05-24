from filters import bilateral
import os

INPUT_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001/'
OUTPUT_FOLDER = '/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/NII/20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_filtered/'
for pathfold,dirs,file_list in os.walk(INPUT_FOLDER):
	for file_name in file_list:
		fpath=os.path.join(pathfold,file_name)
		bilateral.bilateral(fpath,OUTPUT_FOLDER, sig_i=40, sig_g=1.5)

