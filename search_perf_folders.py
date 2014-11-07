import os
import sys

PATH_TO_LOOK = sys.argv[1]
try:
	EXCLUSION_LIST = sys.argv[2]
except:
	EXCLUSION_LIST = None


for p,d,f in os.walk(os.path.join(PATH_TO_LOOK)):
	if len(f)>3000:
		print p


