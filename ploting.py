__author__ = 'ct'
import matplotlib.pyplot as plt
import matplotlib
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
ctr=[100,129,70,9]
RoiSize=20/2
Time,TimeC=curves.samplet()

roi=np.zeros(np.shape(img),dtype='bool')
roi=np.rot90(roi)

roi[ctr[0]-RoiSize:ctr[0]+RoiSize,ctr[1]-RoiSize:ctr[1]+RoiSize,ctr[2]-RoiSize:ctr[2]+RoiSize]=True
ImRoi=np.ma.array(img,mask=roi)

bins=100
low=-200
high=300

TAcurve=np.apply_over_axes(np.average,ImRoi,[0,1,2])
TAcurveSD=np.apply_over_axes(np.std,ImRoi,[0,1,2])

sliceA=img[...,ctr[2],ctr[3]]
sliceS=img[:,ctr[1],:,ctr[3]]
sliceC=img[ctr[0],:,:,ctr[3]]

sliceS=np.rot90(sliceS,3)
sliceC=np.rot90(sliceC,3)

roiA=roi[...,ctr[2],ctr[3]]
roiS=roi[:,ctr[1],:,ctr[3]]
roiC=roi[ctr[0],:,:,ctr[3]]

roiS=np.rot90(roiS,3)
roiC=np.rot90(roiC,3)

hist=scipy.stats.histogram(sliceA,bins,(low,high))


print np.shape(TAcurve[0,0,0])



#Potting
fig=plt.figure(figsize=(18,10))

adj=plt.subplots_adjust(hspace=0.05,wspace=0.05)
gs=matplotlib.gridspec.GridSpec(2,3,width_ratios=[1,2],height_ratios=[2,1,1])

sp1=plt.subplot(gs)
#sp1.vlines(np.linspace(low,high,bins),0,hist[0],color='k', linestyles='solid',linewidth=2)
sp1.errorbar(Time,TAcurve[0,0,0],yerr=TAcurveSD[0,0,0],fmt='-or')


spA=fig.add_subplot(322)
spA.set_axis_off()
spA.imshow(sliceA,cmap='gray',clim=(-200,300),origin='image', extent=(7,0,7,0))
spA.contour(roiA,[0],colors='r',alpha=0.8,extent=(7,0,7,0))
#sp2.contourf(sliceA,[300,2000],colors='b',alpha=0.8)

spS=fig.add_subplot(324)
spS.set_axis_off()
spS.imshow(sliceS,cmap='gray',clim=(-200,300),origin='centre',extent=(7,0,5,0),interpolation='quadric')
spS.contour(roiS,[0],colors='r',alpha=0.8,extent=(7,0,5,0))

spC=fig.add_subplot(326)
spC.set_axis_off()
spC.imshow(sliceC,cmap='gray',clim=(-200,300),origin='bottom',extent=(7,0,5,0), interpolation='quadric')
spC.contour(roiC,[0],colors='r',alpha=0.8,extent=(7,0,5,0))
#imgplot=plt.imshow(slice)
#plt.hist(slice,251,range=(-200,300),fc='k', ec='k')
#imgplot.set_clim=(0.0,1)



plt.show()