__author__ = 'denis'
import numpy as np
import os

folder='/media/WORK/_PERF/KOSTYUKOVA O.I. 03.07.1952/4dNifti/slicerTHIN/'
mrxfile='2.tfm'
#os.mknod(folder+'mangoTransform34.txt')


mrxSlicer=open(folder+mrxfile)
mrxSlicer = mrxSlicer.readlines()[3].split()[1:]

mrxMango=[mrxSlicer[i:i+3] for i in [0,3,6]]
for i in mrxSlicer[-3:]:mrxMango[]

print mrxSlicer[0:3]+[mrxSlicer[-3]]

mrxSlicer[0:3]+mrxSlicer[-3]
mrxMango=[]

mrxMango[:,:3]=mrxSlicer[:9].reshape((3,3))
mrxMango[:,3]=mrxSlicer[-3:]
print mrxMango

open(folder+'mangoTransform34.txt','a').write(mrxMango[0])
