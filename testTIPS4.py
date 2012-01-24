__author__ = 'ct'
import tips4
import numpy as np

image=np.arange(6*6*4*10).reshape(6,6,4,10)
print np.shape(image)
dim_c=3
sigG=2
a=4
if a%2 == 0:
    raise NameError('kernel should have odd size!!!')
GausKern=np.ones((dim_c,dim_c,dim_c))
iterArray=np.ones((dim_c,dim_c,dim_c))
for i,val in np.ndenumerate(iterArray):
    GausKern[i]=tips4.gauss_cl(i,sigG)
    print GausKern.shape
