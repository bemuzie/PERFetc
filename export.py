__author__ = 'ct'
import numpy as np
import matplotlib.pyplot as plt

shp=(20,20,20)
cube_side=5
cube_ctr=[10,10,10]
cube=[slice(i-cube_side,i+cube_side) for i in cube_ctr]

array=np.ones(shp)
array[cube]=3


xd,yd,zd=shp
ctr_x,ctr_y,ctr_z=[5,5,5]
# spherical coordinates

x,y,z=np.ogrid[-ctr_x:xd-ctr_x:1.,-ctr_y:yd-ctr_y:1.,-ctr_z:zd-ctr_z:1.]

r=np.sqrt(x**2+y**2+z**2)
ts=np.arccos(z/r)
yx=np.arctan(y/x)
zx=np.arctan(z/x)


mask=np.absolute(np.round_(yx*np.ones(z.shape),2))==0.8
marray=np.ma.masked_array(array,mask)

print np.array(y/x)[...,0]
print mask[10]


sp1=plt.subplot(121)
sp2=plt.subplot(122)
sp1.imshow(marray[9],interpolation='nearest')
sp2.imshow(np.absolute(np.round_(yx*np.ones(z.shape),2))[8],interpolation='nearest')
plt.show()
plt.draw()
