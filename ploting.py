__author__ = 'ct'
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import curves

def PlotImg(img,matrix,RoiCentre,RoiSize,RoiArt,imgName,invertcord=True,TimeSeries=True):
    Time,TimeC=curves.samplet()
    x,y,z,t=np.shape(img)
    Rx,Ry,Rz=np.ogrid[0:x,0:y,0:z]
    if invertcord == True:
        RACx,RACy,RACz=np.array([x,y,z])-np.array(RoiArt)
        RCx,RCy,RCz=np.array([x,y,z])-np.array(RoiCentre)
    RACx,RACy,RACz=np.array(RoiArt)
    RCx,RCy,RCz=np.array(RoiCentre)

    RoiA=np.sqrt((RACx-Rx)**2+(RACy-Ry)**2+(RACz-Rz)**2)>=RoiSize
    RoiA=RoiA[...,np.newaxis]*np.ones((1,1,1,t))
    ImRoi=np.ma.array(img,mask=RoiA)
    TAcurveA=np.ma.average(np.ma.average(np.ma.average(ImRoi,0),0),0)
    TAcurveSDA=ImRoi.std(0).std(0).std(0)
    CurvePar=curves.fitcurve(TAcurveA[:-3],Time[:-3],initial=[5000,4,1,6,30])
    FittedCurveA=curves.logn(TimeC,CurvePar[0],CurvePar[1],CurvePar[2],CurvePar[3],CurvePar[4])
    ArtMax=np.max(FittedCurveA)
    print CurvePar


    Roi=np.sqrt((RCx-Rx)**2+(RCy-Ry)**2+(RCz-Rz)**2)>=RoiSize
    Roi=Roi[...,np.newaxis]*np.ones((1,1,1,t))
    PerfMap=np.zeros((x,y,z))
    ImRoi=np.ma.array(img,mask=Roi)
    """ #Perfusion map calculating
    for xi,yi,zi in np.ndindex(x,y,z):
        if False not in ImRoi[xi,yi,zi]:
            par=curves.fitcurve(ImRoi[xi,yi,zi,:-3],Time[:-3],[2500,3.5,0.5,0,35])
            FC=curves.logn(TimeC,par[0],par[1],par[2],par[3],par[4])
            maxgrad=np.max(np.gradient(FC))/(TimeC[4]-TimeC[3])
            PerfMap[xi,yi,zi]=6000*maxgrad/ArtMax
            print PerfMap[xi,yi,zi]
            print par
    low=-200
    high=300
    """
    TAcurve=np.ma.average(np.ma.average(np.ma.average(ImRoi,0),0),0)
    TAcurveSD=ImRoi.std(0).std(0).std(0)
    CurvePar=curves.fitcurve(TAcurve[:],Time[:],initial=[3100,3.7,0.75,3,25])
    FittedCurve=curves.logn(TimeC,CurvePar[0],CurvePar[1],CurvePar[2],CurvePar[3],CurvePar[4])

    maxgrad=np.max(np.gradient(FittedCurve))/(TimeC[4]-TimeC[3])

    print CurvePar
    BF=6000*maxgrad/np.max(FittedCurveA)
    print maxgrad,BF

    print TAcurveA.max()

    zsideRatio=matrix[2,2]/matrix[1,1]
    ZRatio=matrix[1,1]/(2*matrix[2,2])
    phase=t-6
    sliceA=img[...,RCz,:]
    sliceA=np.rot90(sliceA)
    sliceS=img[:,RCy,:]
    sliceS=np.rot90(sliceS)
    sliceC=img[RCx,:,:]
    sliceC=np.rot90(sliceC)

    roiA=Roi[...,RCz,:]
    roiA=np.rot90(roiA)
    roiS=(Roi[:,RCy,:])
    roiS=np.rot90(roiS)
    roiC=(Roi[RCx,:,:])
    roiC=np.rot90(roiC)
    """ #Perfusion map calculating
    PerfMapA=PerfMap[RCx-RoiSize:RCx+RoiSize,RCy-RoiSize:RCy+RoiSize,RCz]
    PerfMapA=np.rot90(PerfMapA)
    """
    #Potting
    matplotlib.rc('axes',edgecolor='y',labelcolor='w',labelsize='small',titlesize='medium')
    matplotlib.rc('xtick',color='y')
    matplotlib.rc('ytick',color='y')
    matplotlib.rc('text',color='w')
    fig=plt.figure(facecolor='k')
    plt.subplots_adjust(hspace=0.1,wspace=0)
    fig.suptitle(imgName[48:-4])
    gs=matplotlib.gridspec.GridSpec(2,1,height_ratios=[3,1])

    spImg=matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs[0],width_ratios=[1,ZRatio],wspace=0.05,hspace=0)
    spGraph=matplotlib.gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=gs[1],wspace=0.25)

    spA=plt.subplot(spImg[:,0])
    spA.set_axis_off()


    spC=plt.subplot(spImg[0,1])
    spC.set_axis_off()

    spS=fig.add_subplot(spImg[1,1])
    spS.set_axis_off()

    spTCurve=plt.subplot(spGraph[:])
    """#Perfusion map calculating
    spPerf=plt.subplot(spGraph[0])
    spPerf.set_axis_off()
    """
    spTCurve.set_title('Arterial Blood Flow=%s (ml/min)/100ml'%(round(BF,0)))
    spTCurve.set_xlabel('Time, s')
    spTCurve.set_ylabel("ROI CT density, HU")

    spTCurve.errorbar(Time+12,TAcurve,yerr=TAcurveSD*2,fmt='or')
    spTCurve.plot(TimeC+12,FittedCurve,'-b')
    spTCurve2=spTCurve.twinx()
    spTCurve2.errorbar(Time+12,TAcurveA,yerr=TAcurveSDA*2,fmt='om')
    spTCurve2.plot(TimeC+12,FittedCurveA,'-k')
    spTCurve2.set_ylabel("Aorta CT density, HU")
    """#Perfusion map calculating
    spP=spPerf.imshow(PerfMapA,clim=(0,200))
    fig.colorbar(spP)
    """
    spA.imshow(sliceA[...,phase],cmap='gray',clim=(-200,300),aspect=1,interpolation='bicubic')
    spA.contour(roiA[...,phase],[0],colors='r',alpha=0.8)

    spS.imshow(sliceS[...,phase],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
    spS.contour(roiS[...,phase],[0],colors='r',alpha=0.8)

    spC.imshow(sliceC[...,phase],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
    spC.contour(roiC[...,phase],[0],colors='r',alpha=0.8)

    plt.savefig('/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered/Curves%s.png'%(imgName),facecolor='k')

    if TimeSeries == True:
        print 'true'
        fig2=plt.Figure(1,17)
        gs2=matplotlib.gridspec.GridSpec(1,t)
        for i in np.arange(t):
            print i
            spImg2=matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs2[i],high_ratios=[1,1],wspace=0.05,hspace=0)
            spA2=plt.subplot(spImg2[0,:])
            spA2.set_axis_off()
            spC2=plt.subplot(spImg2[1,0])
            spC2.set_axis_off()
            spS2=fig.add_subplot(spImg2[1,1])
            spS2.set_axis_off()
            spA2.imshow(sliceA[...,i],cmap='gray',clim=(-200,300),aspect=1,interpolation='bicubic')
            spA2.contour(roiA[...,i],[0],colors='r',alpha=0.8)

            spS2.imshow(sliceS[...,i],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
            spS2.contour(roiS[...,i],[0],colors='r',alpha=0.8)

            spC2.imshow(sliceC[...,i],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
            spC2.contour(roiC[...,i],[0],colors='r',alpha=0.8)
        plt.savefig('/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered/Series_%s.png'%(imgName),facecolor='k')
        plt.show()


def out():
#imgplot=plt.imshow(slice)
#plt.hist(slice,251,range=(-200,300),fc='k', ec='k')
#imgplot.set_clim=(0.0,1)

    matplotlib.rc('axes',edgecolor='y',labelcolor='w',labelsize='small',titlesize='medium')
    matplotlib.rc('xtick',color='y')
    matplotlib.rc('ytick',color='y')
    matplotlib.rc('text',color='w')
    fig=plt.figure(facecolor='k')
    plt.subplots_adjust(hspace=0.1,wspace=0)
    fig.suptitle(imgName[48:-4])
    gs=matplotlib.gridspec.GridSpec(2,1,height_ratios=[3,1])

spImg=matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs[0],width_ratios=[1,ZRatio],wspace=0.05,hspace=0)
spGraph=matplotlib.gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=gs[1],wspace=0.25)

spA=plt.subplot(spImg[:,0])
spA.set_axis_off()


spC=plt.subplot(spImg[0,1])
spC.set_axis_off()

spS=fig.add_subplot(spImg[1,1])
spS.set_axis_off()

spTCurve=plt.subplot(spGraph[:])
"""#Perfusion map calculating
spPerf=plt.subplot(spGraph[0])
spPerf.set_axis_off()
"""
spTCurve.set_title('Arterial Blood Flow=%s (ml/min)/100ml'%(round(BF,0)))
spTCurve.set_xlabel('Time, s')
spTCurve.set_ylabel("ROI CT density, HU")

spTCurve.errorbar(Time+12,TAcurve,yerr=TAcurveSD*2,fmt='or')
spTCurve.plot(TimeC+12,FittedCurve,'-b')
spTCurve2=spTCurve.twinx()
spTCurve2.errorbar(Time+12,TAcurveA,yerr=TAcurveSDA*2,fmt='om')
spTCurve2.plot(TimeC+12,FittedCurveA,'-k')
spTCurve2.set_ylabel("Aorta CT density, HU")
"""#Perfusion map calculating
spP=spPerf.imshow(PerfMapA,clim=(0,200))
fig.colorbar(spP)
"""
spA.imshow(sliceA[...,phase],cmap='gray',clim=(-200,300),aspect=1,interpolation='bicubic')
spA.contour(roiA[...,phase],[0],colors='r',alpha=0.8)

spS.imshow(sliceS[...,phase],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
spS.contour(roiS[...,phase],[0],colors='r',alpha=0.8)

spC.imshow(sliceC[...,phase],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
spC.contour(roiC[...,phase],[0],colors='r',alpha=0.8)

plt.savefig('/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered/Curves%s.png'%(imgName),facecolor='k')

if TimeSeries == True:
    print 'true'
    fig2=plt.Figure(1,17)
    gs2=matplotlib.gridspec.GridSpec(1,t)
    for i in np.arange(t):
        print i
        spImg2=matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs2[i],high_ratios=[1,1],wspace=0.05,hspace=0)
        spA2=plt.subplot(spImg2[0,:])
        spA2.set_axis_off()
        spC2=plt.subplot(spImg2[1,0])
        spC2.set_axis_off()
        spS2=fig.add_subplot(spImg2[1,1])
        spS2.set_axis_off()
        spA2.imshow(sliceA[...,i],cmap='gray',clim=(-200,300),aspect=1,interpolation='bicubic')
        spA2.contour(roiA[...,i],[0],colors='r',alpha=0.8)

        spS2.imshow(sliceS[...,i],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
        spS2.contour(roiS[...,i],[0],colors='r',alpha=0.8)

        spC2.imshow(sliceC[...,i],cmap='gray',clim=(-200,300),aspect=zsideRatio,interpolation='bicubic')
        spC2.contour(roiC[...,i],[0],colors='r',alpha=0.8)
    plt.savefig('/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered/Series_%s.png'%(imgName),facecolor='k')
    plt.show()

#imgplot=plt.imshow(slice)
#plt.hist(slice,251,range=(-200,300),fc='k', ec='k')
#imgplot.set_clim=(0.0,1)



