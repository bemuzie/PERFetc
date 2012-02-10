# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
import filters
import matplotlib.pyplot as plt

img=np.ones((50,50,50,1))
img[10:-10,10:-10,10:-10]=5
img[15:-15,15:-15,15:-15]=15
img[20:-20,20:-20,20:-20]=10

img_n=img+np.random.normal(0,5,(50,50,50,1))



img_filtered=filters.bilateralFilter4d(img_n,(1,1,1),1,30)
img_filtered2=filters.bilateralFilter(img_n,7,2,2000)

spI=plt.subplot(231)
spI.imshow(img[...,25,0],clim=(-5,20),interpolation='nearest')
spI.set_title('image')
spB=plt.subplot(232)
spB.imshow(img_filtered[...,25,0],clim=(-5,20),interpolation='nearest')
spB.set_title('bilateral')
spG=plt.subplot(233)
spG.imshow(img_filtered2[...,25,0],clim=(-5,20),interpolation='nearest')
spG.set_title('gausian blur')
spP=plt.subplot(2,3,(4,6))
x_axis=np.arange(len(img_filtered[25,:,25,0]))
spP.plot(x_axis,img_filtered[25,:,25,0],'r-')
spP.plot(x_axis,img_filtered2[25,:,25,0],'b--')
spP.plot(x_axis,img[25,:,25,0])
spP.plot(x_axis,img_n[25,:,25,0])

print np.std(img_n[:10,:10,:10])
print np.std(img_filtered[:10,:10,:10])
print np.std(img_filtered2[:10,:10,:10])
plt.show()

