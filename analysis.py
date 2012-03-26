__author__ = 'denis'
import numpy as np
import matplotlib.pyplot as plt
import image
import matplotlib
folder='/media/WORK/_PERF/TIKHEEV YU.V. 19.02.1935/Nifti4d/'
output_folder='/media/WORK/virtualBox/sharefolder/'

matplotlib.rc('axes',edgecolor='y',labelcolor='w',labelsize='small',titlesize='medium')
matplotlib.rc('xtick',color='y')
matplotlib.rc('ytick',color='y')
matplotlib.rc('text',color='y')
myfig=plt.figure(facecolor='k')
plt.subplots_adjust(hspace=0.1,wspace=0)
fig=plt.subplot(111,axisbg='k')
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
volnum=15.
for i in range(volnum):
    vol='%s.nii'%(i)
    img,hdr,mrx=image.loadnii(folder,vol)
    img[img<0]=-2000
    density=np.average(np.average(img,axis=0),axis=0)
    part=slice(15,-15)
    fig.plot(range(len(density))[part],density[part],color=[i*1/volnum,1-i*1/volnum,0],label=str(i))
leg=plt.legend(loc='upper right')
frame  = leg.get_frame()
frame.set_facecolor('k')

plt.savefig(output_folder+'tikheev.png',facecolor='k')
plt.show()