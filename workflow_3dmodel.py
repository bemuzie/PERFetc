__author__ = 'denis'
import numpy as np
import image
import scipy.ndimage
import skimage.segmentation as skiseg
import skimage.filter as skif
folder='/media/WORK/_PERF/TIKHEEV YU.V. 19.02.1935/Nifti4d/'
folder_out='/media/WORK/_PERF/TIKHEEV YU.V. 19.02.1935/Nifti4d/crop/'
xc,yc,zc=[169,291,94]
print xc
cropsize=60
ROI

for i in range(6,7):
    vol,hdr,mrx=image.loadnii(folder,str(i)+'.nii')
    vol=image.crop(vol[...,None],xc,yc,zc,cropsize)
    vol_out=vol[...,0]
    labels=np.zeros(np.shape(vol_out))
    labels[40,87,54]=1
    labels[82,38,70]=1
    vol_out=skif.tv_denoise(vol_out,200)
    sob=skiseg.w(vol_out,labels)
    #vol_out=scipy.ndimage.median_filter(vol,size=3)
    #vol_out=scipy.ndimage.median_filter(vol_out,size=3)
    """
    sob1=scipy.ndimage.filters.prewitt(vol_out,0)
    sob2=scipy.ndimage.filters.prewitt(vol_out,1)
    sob3=scipy.ndimage.filters.prewitt(vol_out,2)
    sob1=np.absolute(sob1)
    sob2=np.absolute(sob2)
    sob3=np.absolute(sob3)
    sob=sob1+sob2+sob3
    """
    vol_out[vol_out>-15]=1
    vol_out[vol_out<-15]=0

    image.savenii(sob,mrx,folder_out+str(i)+\
                              '_croped_x%s_y%s_z%s_sz%s_2median_sz3_prewitt.nii'\
                              %(xc,yc,zc,cropsize))