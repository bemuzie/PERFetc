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
slice=img[...,70,3]

bins=100
low=-200
high=300



hist=scipy.stats.histogram(slice,bins,(low,high))
print hist

fig=plt.figure()
sp1=fig.add_subplot(211,autoscalex_on=True)
sp1.vlines(np.linspace(low,high,bins),0,hist[0],color='k', linestyles='solid',linewidth=2)

sp2=fig.add_subplot(212)
sp2.set_axis_off()
sp2.imshow(slice,cmap='gray',clim=(-200,300))


#imgplot=plt.imshow(slice)
#plt.hist(slice,251,range=(-200,300),fc='k', ec='k')
#imgplot.set_clim=(0.0,1)
plt.show()