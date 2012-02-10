# -*- coding: utf-8 -*-
__author__ = 'denis'
import numpy as np
import filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes3d
from matplotlib import cm

img=np.ones((50,50,50,1))
img[10:-10,10:-10,10:-10]=5
img[15:-15,15:-15,15:-15]=15
img[20:-20,20:-20,20:-20]=10

img_n=img+np.random.normal(0,5,(50,50,50,1))



img_filtered=filters.bilateralFilter4d(img_n,(1,1,1),1,5)
img_filtered=filters.bilateralFilter4d(img_filtered,(1,1,1),1,5)
img_filtered=filters.bilateralFilter4d(img_filtered,(1,1,1),1,5)

img_filtered2=filters.bilateralFilter4d(img_n,(1,1,1),1,15)

spI=plt.subplot(331)
spI.imshow(img[...,25,0],clim=(-5,20),interpolation='nearest')
spI.set_title('image')
spB=plt.subplot(332)
spB.imshow(img_filtered[...,25,0],clim=(-5,20),interpolation='nearest')
spB.set_title('bilateral')
spG=plt.subplot(333)
spG.imshow(img_filtered2[...,25,0],clim=(-5,20),interpolation='nearest')
spG.set_title('gaussian blur')
spP=plt.subplot(3,3,(4,6))
x_axis=np.arange(len(img_filtered[25,:,25,0]))
spP.plot(x_axis,img_filtered[25,:,25,0],'r-')
spP.plot(x_axis,img_filtered2[25,:,25,0],'b--')
spP.plot(x_axis,img[25,:,25,0])
spP.plot(x_axis,img_n[25,:,25,0])

sp3d1=plt.subplot(3,3,7,projection='3d')
X=np.arange(50)
Y=np.arange(25)
X,Y=np.meshgrid(X,Y)
surf=sp3d1.plot_surface(X,Y,img[0:25,...,25,0],shade=True,rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=True)
sp3d1.view_init(22,20)

sp3d2=plt.subplot(3,3,8,projection='3d')
surf2=sp3d2.plot_surface(X,Y,img_filtered[0:25,...,25,0],shade=True, rstride=1, cstride=1, cmap=cm.jet,linewidth=0.1)
sp3d2.view_init(22,20)


sp3d3=plt.subplot(3,3,9,projection='3d')
surf3=sp3d3.plot_surface(X,Y,img_filtered2[0:25,...,25,0],shade=True, rstride=1, cstride=1, cmap=cm.jet,linewidth=0.1)

sp3d3.view_init(22,20)
print np.std(img_n[:10,:10,:10])
print np.std(img_filtered[:10,:10,:10])
print np.std(img_filtered2[:10,:10,:10])
plt.show()

