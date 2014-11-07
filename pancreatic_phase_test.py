__author__ = 'denest'

from perfusion.express import *
import numpy as np
import matplotlib.pyplot as plt

a = [44.64877990864637,
 60.49938896933365,
 185.7885957985045,
 424.99678292042654,
 521.9514543622375,
 473.2310834657947,
 325.303563042722,
 175.04206749112657,
 108.29381960598403,
 85.55330475432997,
 83.41730958452371,
 125.78528484932369,
 114.54445596352669,
 113.53738966541108,
 116.15463139449018,
 117.12762712371725,
 105.83024818019554,
 101.09165687256322,
 94.59840815055142]
a=np.array(a)
a2 = [44.64877990864637,
 60.49938896933365,
 185.7885957985045,
 424.99678292042654,
 521.9514543622375,
 473.2310834657947,
 465.303563042722,
 475.04206749112657,
 408.29381960598403,
 385.55330475432997,
 283.41730958452371,
 325.78528484932369,
 414.54445596352669,
 513.53738966541108,
 216.15463139449018,
 217.12762712371725,
 105.83024818019554,
 101.09165687256322,
 94.59840815055142]
t = np.array([10,
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
a = np.array(a)
a -= a[0]
print len(t)
new_times = np.arange(t[0], t[-1] + 1)
input_tac_s, input_tac_pars = spline_interpolation(a, t, new_times)



simple_tac = calc_tissue_tac(input_tac_pars, 10, 0.3, t)
#simple_tac2 = calc_tissue_tac_conv(input_tac_s, 10, 0.3, new_times, t, 'trap')
simple_tac3 = calc_tissue_tac_conv(input_tac_s, 14, 0.4, new_times, t, 'lognorm', 1)
simple_tac2 = calc_tissue_tac_conv(input_tac_s, 19, 0.35, new_times, t, 'lognorm', 0.81)
simple_tac4 = calc_tissue_tac_conv(input_tac_s, 24, 0.35, new_times, t, 'lognorm', 0.41)
pancreatic =  np.where(simple_tac3==np.max(simple_tac3))[0]


tacs=[]
bad_tacs=[]
maximums = np.zeros(t.shape[0])

p_phase = np.where(simple_tac4==np.max(simple_tac4))[0]
print p_phase

#plt.plot(t,a)
#plt.plot(t,simple_tac4)
pars_subset = perf_pars_gen(bvs=np.arange(0.01,0.2,0.05),
                            mtts=np.arange(10,100,10),
                            sigmas=np.arange(0.1,2,0.5),
                            lags=(0,))
print pars_subset
calc_tissue_tacs_mrx(a, pars_subset, new_times, t_subset=None)



