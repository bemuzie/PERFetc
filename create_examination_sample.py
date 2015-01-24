__author__ = 'denest'
import sys
sys.path.insert(0, '/home/denest/PERFetc2/')
import matplotlib.pyplot as plt
import numpy as np
import workflow
from perfusion import express
import os
def lag_input(input_curve,LAG=0,scale=1):
    return np.append(np.repeat(input_curve[0],LAG),input_curve)[:input_curve.shape[0]]*scale

#set const
ROOT_FOLDER_LIN = '/media/WORK_/_PERF/KUZNETSOVA G.P. 22.09.1946/20140617_641'

wf = workflow.Workflow(ROOT_FOLDER_LIN)
wf.dir_manager.add_path('FILTERED', 'filtered', add_to='nii')
wf.setup_env(mricron='/home/denest/mricron/dcm2nii')
#wf.make_time_file()
#wf.convert_dcm_to_nii(make_time=True)
#wf.separate_nii()
#wf.filter_vols(intensity_sigma=40, gaussian_sigma=1.5)
wf.update_label()

AORTA = np.array(wf.rois.get_concentrations('aorta'))
print AORTA
times=np.array(wf.rois.get_time_list())
print times
"""
AORTA = np.array([44.64877990864637,
         60.49938896933365,
         185.7885957985045,
         424.99678292042654,
         521.9514543622375,
         473.2310834657947,
         355.303563042722,
         245.04206749112657,
         208.29381960598403,
         185.55330475432997,
         183.41730958452371,
         125.78528484932369,
         114.54445596352669,
         113.53738966541108,
         116.15463139449018,
         117.12762712371725,
         105.83024818019554,
         101.09165687256322,
         94.59840815055142])
"""
"""
AORTA = np.array([44.64877990864637,
         60.49938896933365,
         185.7885957985045,
         424.99678292042654,
         424.99678292042654,
         424.99678292042654,
         424.99678292042654,
         324.99678292042654,
         124.99678292042654,
         124.99678292042654,
         224.99678292042654,
         224.99678292042654,
         224.99678292042654,
         144.99678292042654,
         124.99678292042654,
         124.99678292042654,
         105.83024818019554,
         101.09165687256322,
         94.59840815055142])

times = np.array([10,
                  13,
                  15,
                  17,
                  20,
                  22,
                  24,
                  26,
                  29,
                  31,
                  33,
                  42,
                  47,
                  52,
                  57,
                  62,
                  91,
                  101,
                  111])
"""
#make input tac
new_time_steps = np.arange(times[0], times[-1] + 1, 1)
input_tac_smoothed, input_tac_splines = express.spline_interpolation(AORTA, times, new_time_steps)
time_steps = np.arange(10,60,2)
#lag and scale input
input_tac_smoothed = lag_input(input_tac_smoothed)
input_tac_smoothed-=input_tac_smoothed[0]

pars_subset = express.perf_pars_gen(bvs=np.arange(0.01,0.7,0.1),
                                    mtts=np.arange(2,40,5),
                                    sigmas=np.arange(0.1,20,10),
                                    lags=(0,),
                                    bvs2=(0,),
                                    mtts2=(0,),
                                    sigmas2=(0,),
                                    lags2=(0,))
pancreatic_pars = np.array([[1.2,0.7,0,81,0,0,0,0],
                            [9,0.28,0,16,0,0,0,0],
                            [6,0.41,0,16,0,0,0,0],
                            [11,0.25,0,11,0,0,0,0],
                            [9,0.33,0,23,0,0,0,0],
                            [1,0.41,0,15,0,0,0,0],
                            [7,0.29,0,27,0,0,0,0]
                           ])
tumor_pars = np.array([[20,0.15,0,31,0,0,0,0],])


pancreatic_tac = express.calc_tissue_tac_from_pars(input_tac_smoothed,
                                          time_steps=new_time_steps,
                                          time_subset=time_steps,
                                          params=pancreatic_pars )
tumor_tac = express.calc_tissue_tac_from_pars(input_tac_smoothed,
                                          time_steps=new_time_steps,
                                          time_subset=time_steps,
                                          params=tumor_pars )


plt.plot(times,AORTA/10.)
#plt.plot(new_time_steps,input_tac_smoothed/10.)
#print AORTA>100
#print np.where(AORTA>100)[0][0]
print pancreatic_tac[0]
#plt.plot(new_time_steps,pancreatic_tac,alpha=0.3)
plt.plot(time_steps,pancreatic_tac[:,0],'g')
#plt.plot(new_time_steps,pancreatic_tac[:,1],'b')
#plt.plot(time_steps,tumor_tac,'k')
diff_tac=pancreatic_tac-tumor_tac
#plt.plot(new_time_steps,diff_tac,'--',alpha=.5)
plt.plot(time_steps,diff_tac[:,0],'g--')
#plt.plot(new_time_steps,diff_tac[:,1],'b--')
#plt.plot(time_steps,diff_tac[:,2],'k--')


plt.plot(time_steps,tumor_tac,'r')
#plt.plot(new_time_steps,tumor_tac,'k--')

#plt.plot(times,wf.rois.get_concentrations('pancreas'),'k')
treshold100=times[np.where(AORTA>100)[0][[0]]]
print treshold100
plt.axvline(treshold100+15,0)
plt.axvline(treshold100+20,0)
plt.axvline(40,0)
plt.axvline(60,0)
plt.axhline(10)
plt.show()

print tumor_tac-pancreatic_tac[0]
#push to table
"""
CSV_PATH = '/home/denest/DISSER/R/data/ts_hu__test.csv'

rois={'aorta':AORTA,
      'pancreas':np.ravel(pancreatic_tac),
      'tumor':np.ravel(tumor_tac)}

p_id='obukhova_kovalchuk_test'
with open(os.path.join(CSV_PATH),'a') as summary_csv:
      for r_name,r_tac in rois.items():
            for t,rv in zip(times,r_tac):
                  summary_csv.write(','.join(map(str,[p_id,t,r_name,rv,'\n'])))
"""
"""
CSV_PATH = '/home/denest/DISSER/R/data/ts_hu.csv'
AORTA_SMOOTHED = lag_input(express.spline_interpolation(AORTA, times, time_steps)[0])

rois={'aorta':np.round(AORTA_SMOOTHED),
      'pancreas':np.round(np.ravel(pancreatic_tac[:,0]) + np.random.randint(40,60)),
      'tumor':np.round(np.ravel(tumor_tac)+ np.random.randint(40,60))}

p_id='kuznecova_test2'
with open(os.path.join(CSV_PATH),'a') as summary_csv:
      for t in enumerate(time_steps):
            list_to_write = [p_id,t[1],rois['aorta'][t[0]],rois['pancreas'][t[0]],rois['tumor'][t[0]]]
            summary_csv.write(','.join(map(str,list_to_write ))+'\n')
"""