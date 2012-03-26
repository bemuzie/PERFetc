# -*- coding: utf-8 -*-
from curves import curves

__author__ = 'denis'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import image

img,header,mrx=image.loadnii('/media/WORK/_PERF/SILAGI A.L. 23.05.1958/Nifti4d',\
                             'GeneralBodyPerfusionSILAGIAL23051958s004a001_FC17QDS.nii')

output_folder='/media/WORK/_PERF/SILAGI A.L. 23.05.1958/pics/'

rois=dict(
    artery=(258,239,103,10),
    cyst=(221,277,136,5),
    head=(235,282,190,5),
    tail=(318,253,239,5)
)

roisdata=dict([(i, curves.Roi(img,rois[i][0:-1],rois[i][-1],'sphere',
                             True,[mrx[1,1],mrx[1,1],mrx[2,2]],1,1500,11,1)) for i in rois])
time,timec= curves.samplet()
roisdata['cyst'].fitcurve(time,[1500,3,0.6,6,0],stop=16)
roisdata['tail'].fitcurve(time,[1500,3,0.6,6,40],stop=16)
roisdata['head'].fitcurve(time,[2670,3,0.8,6,40],stop=16)
roisdata['artery'].fitcurve(time,[2670,3,0.8,2,10],stop=14)

print roisdata['artery'].tac

sideratio=mrx[2,2]/mrx[1,1]

fig=plt.figure()
plt.subplots_adjust(hspace=0.1,wspace=0)
for i in roisdata:
    matplotlib.rc('text',color='w')
    sp1= plt.subplot(111)
    sp1.set_title('unfiltered')
    sp1.set_axis_off()
    for axis in roisdata[i].center:
        if axis == 'z':
            plt.delaxes()
            sp1= fig.add_subplot(111)
            sp1.set_title('%s'%i)
            sp1.set_axis_off()
            sp1.imshow(roisdata[i].sliceAx,cmap='gray',clim=(-120,280),aspect=1,interpolation='bicubic')
            plt.savefig(output_folder+i+'_'+axis+'_imgonly.png',facecolor='k')
            sp1.axvline(x=roisdata[i].center['x'])
            sp1.axhline(y=512-roisdata[i].center['y'])
            plt.savefig(output_folder+i+'_'+axis+'.png',facecolor='k')
        if axis == 'x':
            plt.delaxes()
            sp1= plt.subplot(111)
            sp1.set_title('%s'%i)
            sp1.set_axis_off()
            sp1.imshow(roisdata[i].sliceCor,cmap='gray',clim=(-120,280),aspect=sideratio,interpolation='bicubic')
            plt.savefig(output_folder+i+'_'+axis+'_imgonly.png',facecolor='k')
            sp1.axvline(x=roisdata[i].center['x'])
            sp1.axhline(y=len(img[0,0])-roisdata[i].center['z'])
            plt.savefig(output_folder+i+'_'+axis+'.png',facecolor='k')
        if axis == 'y':
            plt.delaxes()
            sp1= plt.subplot(111)
            sp1.set_title('%s'%i)
            sp1.set_axis_off()
            sp1.imshow(roisdata[i].sliceSag,cmap='gray',clim=(-120,280),aspect=sideratio,interpolation='bicubic')
            plt.savefig(output_folder+i+'_'+axis+'_imgonly.png',facecolor='k')
            sp1.axvline(x=roisdata[i].center['y'])
            sp1.axhline(y=len(img[0,0])-roisdata[i].center['z'])
            plt.savefig(output_folder+i+'_'+axis+'.png',facecolor='k')




plt.delaxes()
matplotlib.rc('axes',edgecolor='y',labelcolor='w',labelsize='small',titlesize='medium')
matplotlib.rc('xtick',color='y')
matplotlib.rc('ytick',color='y')
matplotlib.rc('text',color='w')
spTCurve=plt.subplot(111)
#spTCurve.set_title('Arterial Blood Flow=%s (ml/min)/100ml'%(round(BF,0)))
phead="Головка поджелудочной железы"

headP=spTCurve.errorbar(time+12,roisdata['head'].tac,yerr=roisdata['head'].tacsd*2,fmt='or')
tailP=spTCurve.errorbar(time+12,roisdata['tail'].tac,yerr=roisdata['tail'].tacsd*2,fmt='ob')
cystP=spTCurve.errorbar(time+12,roisdata['cyst'].tac,yerr=roisdata['cyst'].tacsd*2,fmt='ok')

headC=spTCurve.plot(timec+12,roisdata['head'].tacfit,'-r')
tailC=spTCurve.plot(timec+12,roisdata['tail'].tacfit,'-b')


spTCurve2=spTCurve.twinx()
arteryP=spTCurve2.errorbar(time+12,roisdata['artery'].tac,yerr=roisdata['artery'].tacsd*2,fmt='om')
arteryC=spTCurve2.plot(timec+12,roisdata['artery'].tacfit,'-m')
spTCurve2.set_ylabel("Aorta CT density, HU")
spTCurve.set_xlabel('Time, s')
spTCurve.set_ylabel("ROI CT density, HU")


BloodFlowHead=round(6000*np.max(np.gradient(roisdata['head'].tacfit))/np.max(roisdata['artery'].tacfit),2)
BloodFlowTail=round(6000*np.max(np.gradient(roisdata['tail'].tacfit))/np.max(roisdata['artery'].tacfit),2)

spTCurve.text(0.9,0.1,'Arterial Blood Flow=%s (ml/min)/100ml'%BloodFlowHead,
    horizontalalignment='right',
    verticalalignment='center',
    transform = spTCurve.transAxes,
    bbox=dict(facecolor='r', alpha=0.8))
spTCurve.text(0.9,0.05,'Arterial Blood Flow=%s (ml/min)/100ml'%BloodFlowTail,
    horizontalalignment='right',
    verticalalignment='center',
    transform = spTCurve.transAxes,
    bbox=dict(facecolor='b', alpha=0.8))


leg=spTCurve.legend([headC,tailC,arteryC],['Caput pancreatis',"Cauda pancreatis","Aorta"],'upper right')
frame  = leg.get_frame()
frame.set_facecolor('k')

plt.savefig(output_folder+'curves.png',facecolor='k')
print roisdata['head'].pars
print roisdata['tail'].pars

