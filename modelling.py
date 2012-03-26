# -*- coding: utf-8 -*-
from curves import curves

__author__ = 'denis'
import matplotlib.pyplot as plt
tr,tc = curves.samplet(11.,2.,6.,4.)
#making passage curve
tiss,tiss_tc= curves.passcurve_l(tr,3000.,3.,0.6,ts=10.,tc=tc,b=50.)
tissS,tissS_tc= curves.passcurve_l(tr,1000.,3.5,0.6,ts=20.,tc=tc,b=0.)
tissR_tc=tiss_tc+tissS_tc
tissR=tiss+tissS
artflow,artflow_tc= curves.passcurve_l(tr,4000.,2.,0.75,ts=3.,tc=tc,b=40.)

tissn= curves.modelfit(tissR,tiss_tc,tr,it=100,ns=5,name='tiss')
artflown= curves.modelfit(artflow,artflow_tc,tr,it=100,ns=10,name='art')


#plot passage curves
plt.plot(tc,tiss_tc,'-r',tr,tissn,'o-g',tc,tissR_tc,'r')

plt.plot(tc,artflow_tc,'-r',tr,artflown,'o-g')
#print(maxgrad,maxgrad_f)
#print (popt)
plt.show()


