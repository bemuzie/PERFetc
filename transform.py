__author__ = 'denis'
import numpy as np
import os


folder='/media/63A0113C6D30EAE8/_PERF/YAVNIK/slicer/'
mrxfile='stack.tfm'
mrxfileMango=''
mrxfileMangoAbs=folder+mrxfileMango
outputim=''
inputim=''

os.mknod(mrxfileMangoAbs)



mrxSlicer=open(folder+mrxfile)
mrxSlicer = mrxSlicer.readlines()[3].split()[1:]

mrxMango=[mrxSlicer[i:i+3] for i in [0,3,6]]
mrxMango=[mrxMango[i-1]+[mrxSlicer[-i]] for i in [1,2,3]]


for i in mrxMango:
        open(mrxfileMangoAbs,'a').write(' '.join(i)+'\n')

os.system('applytransform -c %s %s %s'%(mrxfileMangoAbs,inputim,outputim))
