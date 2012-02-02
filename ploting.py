__author__ = 'ct'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image
import os
import scipy.stats
import curves

adress="/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered"
filelist=os.listdir(adress)


for file in filelist[:1]:
    print file
    vol,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(vol)
#50,50,70
print mrx
img=vol
img=np.rot90(img)




"""
img=np.rot90(img,3)
img=np.swapaxes(img,2,0)
"""
ctr=[55,129,70,9]
RoiSize=20/2
Time,TimeC=curves.samplet()

bins=100
low=-200
high=300


sliceA=img[...,ctr[2],ctr[3]]
sliceS=np.swapaxes(img[:,ctr[1],:,ctr[3]],0,1)
sliceC=np.swapaxes(img[ctr[0],:,:,ctr[3]],0,1)

roi=np.zeros(np.shape(img),dtype='bool')
roi[ctr[0]-RoiSize:ctr[0]+RoiSize,ctr[1]-RoiSize:ctr[1]+RoiSize,ctr[2]-RoiSize:ctr[2]+RoiSize]=True
roiA=roi[...,ctr[2],ctr[3]]
roiS=np.swapaxes(roi[:,ctr[1],:,ctr[3]],0,1)
roiC=np.swapaxes(roi[ctr[0],:,:,ctr[3]],0,1)
ImRoi=img[ctr[0]-RoiSize:ctr[0]+RoiSize,ctr[1]-RoiSize:ctr[1]+RoiSize,ctr[2]-RoiSize:ctr[2]+RoiSize]

hist=scipy.stats.histogram(sliceA,bins,(low,high))
TAcurve=np.apply_over_axes(np.average,ImRoi,[0,1,2])
TAcurveSD=np.apply_over_axes(np.std,ImRoi,[0,1,2])
print np.shape(TAcurve[0,0,0])

fig=plt.figure(figsize=(10,10))
adj=plt.subplots_adjust(hspace=0.05,wspace=0.05)

sp1=fig.add_subplot(3,2,(1,5))
#sp1.vlines(np.linspace(low,high,bins),0,hist[0],color='k', linestyles='solid',linewidth=2)
sp1.errorbar(Time,TAcurve[0,0,0],yerr=TAcurveSD[0,0,0],fmt='-or')

sp2=fig.add_subplot(322)
sp2.set_axis_off()
sp2.imshow(sliceA,cmap='gray',clim=(-200,300))
sp2.contour(roiA,[0],colors='r',alpha=0.8)
#sp2.contourf(sliceA,[300,2000],colors='b',alpha=0.8)

sp4=fig.add_subplot(324)
sp4.set_axis_off()
sp4.imshow(sliceS,cmap='gray',clim=(-200,300),origin='centre',extent=(70,0,50,0),interpolation='quadric')
sp4.contour(roiS,[0],colors='r',alpha=0.8,extent=(0,70,0,50))

sp5=fig.add_subplot(326)
sp5.set_axis_off()
sp5.imshow(sliceC,cmap='gray',clim=(-200,300),origin='centre',extent=(7,0,5,0), interpolation='quadric')
sp5.contour(roiC,[0],colors='r',alpha=0.8,extent=(0,7,0,5))
#imgplot=plt.imshow(slice)
#plt.hist(slice,251,range=(-200,300),fc='k', ec='k')
#imgplot.set_clim=(0.0,1)



plt.show()