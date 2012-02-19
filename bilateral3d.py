__author__ = 'denis'

def bilateralFilter(img,voxel_size,sigg,sigi):
    """ 4d Bilateral exponential filter.
    image array, kernel size, distance SD, intensity SD
    """
    #img=np.array(img,dtype=float)
    sigISqrDouble=float(sigi*sigi*2)
    gkern=gauss_kernel_3d(sigg, voxel_size)
    ksize=np.shape(gkern)
    GausKern=np.ravel(gkern)
    #Closness function
    kwargs=dict(sigISqrDouble=sigISqrDouble,GausKern=GausKern,center=len(GausKern)/2)
    img_filtered=ndimage.generic_filter(img,bilateralFunc,size=ksize+(1,),extra_keywords=kwargs)
    return img_filtered
