__author__ = 'denest'
import matplotlib.pyplot as plt
from perfusion import express as expr
import numpy as np
#Simulation pars
times = [10,
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
                  111]
input_tac = [42.6931152344	,
45.2479934692	,
46.4631233215	,
92.3175735474	,
209.150650024	,
272.341552734	,
333.351257324	,
439.620269775	,
464.547088623	,
382.668914795	,
275.615081787	,
117.078765869	,
144.830245972	,
191.995788574	,
178.131362915	,
153.070007324	,
136.892059326	,
125.342666626	,
121.510551453	
]
input_tac2 = [27.1931705475	,
28.6477222443	,
26.8838615417	,
28.2938919067	,
28.6249599457	,
29.9340991974	,
35.0015144348	,
41.7257919312	,
56.3434867859	,
78.5413665771	,
108.760818481	,
188.567642212	,
160.314804077	,
137.499481201	,
124.160339355	,
130.748962402	,
115.376441956	,
106.52596283	,
100.415275574]

#subtract tacs
input_tac = np.array(input_tac)-input_tac[0]
input_tac2 = np.array(input_tac2)-input_tac2[0]
#

BV_range_a = np.arange(0.01,0.99,0.4)
BV_range_p = (0.01,1,0.1)

MTT_range_a = np.arange(0.1,100,30)
MTT_range_p = (0,100,1)

S_range_a = np.arange(0.01,2,0.5)
S_range_p = (0.01,2,0.1)

#Simulation
new_time_steps = np.arange(times[0], times[-1] + 1, 1)

input_tac_smoothed, input_tac_splines = expr.spline_interpolation(input_tac, times, new_time_steps)
input_tac_smoothed2, input_tac_splines2 = expr.spline_interpolation(input_tac2, times, new_time_steps)

input_tac_smoothed2[new_time_steps < times[0]] = input_tac2[0]
input_tac_smoothed[new_time_steps < times[0]] = input_tac[0]


reference_tacs, reference_pars = expr.calc_tissue_tac_mrx_conv2(input_tac_smoothed,
                                                                    input_tac_smoothed2,
                                                                    time_steps=new_time_steps,
                                                                    time_subset=times,
                                                                    rc_type='lognorm',
                                                                    mtts=MTT_range_a,
                                                                    bvs=BV_range_a,
                                                                    rc_sigma=S_range_a,
                                                                    lags=[0])

#plots


reference_tacs = expr.calc_tissue_tac_from_pars(input_tac_smoothed,
                                                input_tac_smoothed2,
                                                time_steps=new_time_steps,
                                                time_subset=times,
                                                rc_type='lognorm',
                                                params = reference_pars)

plt.plot(times,input_tac2)
plt.plot(new_time_steps,input_tac_smoothed2)
plt.plot(times,input_tac)
plt.plot(new_time_steps,input_tac_smoothed)
plt.show()


for tac,pars in zip(reference_tacs, reference_pars):
	print pars
	plt.plot(times,tac,'o-',color=(pars[1],1,pars[5]),alpha=0.5)
plt.show()