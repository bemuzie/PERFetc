__author__ = 'ct'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image
import os
import scipy.stats

adress="/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered"
filelist=os.listdir(adress)


for file in filelist[:1]:
    print file
    img,hdr, mrx=image.loadnii(adress,file)
    shp=np.shape(img)
#50,50,70
sliceA=img[...,70,9]
sliceS=img[:,100,:,9]
sliceC=img[80,:,:,9]
bins=100
low=-200
high=300




hist=scipy.stats.histogram(sliceA,bins,(low,high))
print hist

fig=plt.figure(figsize=(10,10))
adj=plt.subplots_adjust(hspace=0.05,wspace=0.05)

sp1=fig.add_subplot(2,2,(1,3))
sp1.vlines(np.linspace(low,high,bins),0,hist[0],color='k', linestyles='solid',linewidth=2)

sp2=fig.add_subplot(222)
sp2.set_axis_off()
sp2.imshow(sliceA,cmap='gray',clim=(-200,300))
sp2.contourf(sliceA,[-1000,-200],colors='r',alpha=0.8)
sp2.contourf(sliceA,[300,2000],colors='b',alpha=0.8)
sp4=fig.add_subplot(224)
sp4.set_axis_off()
sp4.imshow(sliceS,cmap='gray',clim=(-200,300),origin='centre',extent=(0.5,0,0.7,0),interpolation='quadric')
#imgplot=plt.imshow(slice)
#plt.hist(slice,251,range=(-200,300),fc='k', ec='k')
#imgplot.set_clim=(0.0,1)



plt.show()