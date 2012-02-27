__author__ = 'ct'
import numpy as np
import timeit
import filters
import ndbilateral
import matplotlib.pyplot as plt
from scipy import ndimage

imgxy=100
imgz=50
imgt=1

img=np.ones((imgxy,imgxy,imgz,imgt))
contour1=[slice(17,-17) for i in [0,1,2]]
contour2=[slice(20,-20) for i in [0,1,2]]
contour3=[slice(30,-30) for i in [0,1,2]]

img[contour1]=-50
img[contour2]=150
img[contour3]=140

img_n=img+np.random.normal(0,80,(imgxy,imgxy,imgz,imgt))


voxel_size=[1,1,1]
gauss_sigma=2
intensity_sigma=10000
num=1

ROI=[[0,None],[0,None],[0,None],[0,None]]


#a=filters.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)
b=ndbilateral.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)
b=ndbilateral.bilateral(b,voxel_size,1,60)
img_diff=img_n-b

roi=contour3.append(slice(3,4))
snr=np.average(b[roi])/np.std(b[roi])
newsnr_prew=0
print snr
for i in np.linspace(1,6,1000):
    b=img_n-i*img_diff
    newsnr=np.average(b[roi])/np.std(b[roi])

    print newsnr, 'snr:',snr, newsnr_prew
    if newsnr<snr or newsnr<newsnr_prew:
        break
    newsnr_prew=newsnr
img_median=ndimage.generic_filter(b,np.median,(3,3,3,1))
"""
t = timeit.Timer("filters.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)",
    "from __main__ import *")
t2 = timeit.Timer("ndbilateral.bilateral(img_n,voxel_size,gauss_sigma,intensity_sigma)",
    "from __main__ import *")
print "Pure python function", t.timeit(num), "sec"
print "Cython function", t2.timeit(num), "sec"
"""

images=dict(
    image=img_median,
    noisy=img_n,
    diff_image_cython=img_median-img,
    cython=b,
    diff_noise_cython=img_median-img_n,
    diff_image_noise=img-img_n
)
i=1
for title,pic in images.items():
    sp=plt.subplot(2,3,i)
    sp.set_title(title)
    sp.imshow(pic[-50:,-50:,imgz//2,0],clim=(-50,200))
    i+=1

plt.show()

