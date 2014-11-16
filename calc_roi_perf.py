__author__ = 'denest'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from perfusion import express

def create_tacs(input_tac, times):
    new_time_steps = np.arange(times[0], times[-1] + 1, 1)
    input_tac_smoothed, input_tac_splines = express.spline_interpolation(input_tac, times, new_time_steps)
    pars_subset = express.combine_pars(bv=np.arange(0.1,0.60,0.1),
                                       a=np.arange(1,15,0.5),
                                       loc=np.arange(0.1,13,0.3),
                                       scale=np.arange(0.1,6,0.3))
    tacs = express.calc_tissue_tacs_mrx_3gamma(input_tac_smoothed, params = pars_subset, time_steps = new_time_steps, time_subset = times)
    return tacs,pars_subset

if __name__ == '__main__':
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
    print a
    aorta_tac = np.array(a)

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
    pancreas = [50,
                50,
                61,
                75,
                94,
                109,
                115,
                122,
                110,
                106,
                102,
                88,
                83,
                84,
                80,
                81.06878662,
                76.95361328,
                77.53733063,
                77.64296722]
    print t.shape, len(pancreas)

    pancreas_tac = np.array(pancreas)
    pancreas_tac -= pancreas_tac[0]
    tacs,pars_subset = create_tacs(aorta_tac, t)

    diff=tacs-pancreas_tac[:,None]
    diff*=diff
    diff=np.sum(diff,axis=0)
    ssd_threshold=np.sort(diff)[50]
    #print np.where(diff< ssd_threshold),tacs.shape
    #print pars_subset[np.where(diff< ssd_threshold)[0]][:,3]
    plt.subplot(211)
    plt.plot(t,tacs[:,np.where(diff< ssd_threshold)[0]],'ko-',alpha=0.3)
    plt.plot(t,pancreas_tac,'go-')
    plt.subplot(212)
    
    print pars_subset
    slice_subset=np.where(diff< ssd_threshold)[0]
    print pars_subset['a'][slice_subset],'\n', pars_subset['loc'][slice_subset],'\n', pars_subset['scale'][slice_subset]
    

    times_for_gamma=np.arange(0,100,1)[...,None].repeat( np.where(diff< ssd_threshold)[0].shape[0] ,axis=1)
    #print times_for_gamma.shape
    pdfs=stats.gamma.pdf(times_for_gamma, 
                        pars_subset['a'][slice_subset],
                        loc=pars_subset['loc'][slice_subset], 
                        scale=pars_subset['scale'][slice_subset])
    print 'mtt',stats.gamma.mean(pars_subset['a'][slice_subset],
                        loc=pars_subset['loc'][slice_subset],
                        scale=pars_subset['scale'][slice_subset])
    print 'bv',pars_subset['bv'][slice_subset]

    #print times_for_gamma.shape,pdfs.shape
    plt.plot(np.arange(0,100,1)[...,None], pdfs)


    plt.show()
