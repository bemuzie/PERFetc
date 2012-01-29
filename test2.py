__author__ = 'ct'
"""
from image import dcm_parser
fromfolder='/media/WORK/_PERF/temp/270112_20120127_173903504/'
tofolder='/media/WORK/_PERF/'
dcm_parser(fromfolder,tofolder)
"""
import gdcm
folder='/media/WORK/_PERF/temp/270112_20120127_173903504/'

tag1=gdcm.Tag(0x8,0x8)
tag2=gdcm.Tag(0x10,0x10)


gdcmdir=gdcm.Directory()
nfiles=gdcmdir.Load(folder)

print (nfiles)