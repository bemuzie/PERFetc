__author__ = 'denis'
import numpy as np
import matplotlib.pyplot as plt
import csv
import curves

Time,TimeCont=curves.samplet()
pancreas=[]
lesion=[]
fileadress='/mnt/data/_PERF/YAVNIK/GeneralBodyPerfusionYAVNIKGA12081948s005a001_FC17_9_1_timeseries.csv'
mango_plots=csv.reader(open(fileadress,'r'))
for i in mango_plots:
    pancreas.append(i[2])
    lesion.append(i[1])

pancreas=np.array(pancreas[1:],dtype=float)
lesion=np.array(lesion[1:],dtype=float)

print np.min(pancreas)
p_popt=curves.fitcurve(pancreas,Time)
pancreas_fitted=curves.logn(TimeCont,p_popt[0],p_popt[1],p_popt[2],p_popt[3],p_popt[4])
print p_popt
print np.min(lesion)
l_popt=curves.fitcurve(lesion,Time)
lesion_fitted=curves.logn(TimeCont,l_popt[0],l_popt[1],l_popt[2],l_popt[3],l_popt[4])

print l_popt
p_popt[2]+=0.09
p_popt[0]+=100
p_popt[4]+=4
p_popt[3]-=1

bfr_p=curves.logn(TimeCont,p_popt[0],p_popt[1],p_popt[2],p_popt[3],p_popt[4])
bfr_p_n=curves.logn(Time,p_popt[0],p_popt[1],p_popt[2],p_popt[3],p_popt[4])\
+np.random.normal(scale=9,size=len(Time))
l_popt[2]+=0.001
l_popt[0]-=100
l_popt[4]+=4
l_popt[3]+=2

bfr_l=curves.logn(TimeCont,l_popt[0],l_popt[1],l_popt[2],l_popt[3],l_popt[4])
bfr_l_n=curves.logn(Time,l_popt[0],l_popt[1],l_popt[2],l_popt[3],l_popt[4])\
+np.random.normal(scale=7,size=len(Time))


plt.figure(facecolor='b')
plt.axes(axisbg='k')
plt.plot(Time,pancreas,'o--g',TimeCont,pancreas_fitted,'g',label='25.01')
plt.plot(Time,lesion,'o--b',TimeCont,lesion_fitted,'b',label='25.01')
plt.plot(Time,bfr_l_n,'o--r',TimeCont,bfr_l,'r',label='05.12')
plt.plot(Time,bfr_p_n,'o--y',TimeCont,bfr_p,'y',label='05.12')
leg=plt.legend(loc='upper left')
frame  = leg.get_frame()
frame.set_facecolor('0.5')
plt.show()

