__author__ = 'denest'
import sys
import os

sys.path.insert(0, '/home/denest/PERFetc2/')
import matplotlib.pyplot as plt
import numpy as np
import workflow
from perfusion import express
from scipy import stats


def getbest(tacs, reference, best_num=1):
    diff = tacs - reference[:, None]
    diff *= diff
    diff = np.sum(diff, axis=0)
    ssd_threshold = np.sort(diff)[best_num]
    return np.where(diff < ssd_threshold)[0]


# choose_tacs

if __name__ == "__main__":
    ROOT_FOLDER_LIN = './'
    wf = workflow.Workflow(ROOT_FOLDER_LIN)
    wf.update_label()
    #get time steps
    times = wf.rois.get_time_list()
    times_all = np.arange(times[0], times[-1] + 1, 1)
    #get TACs
    pancreas_tac = np.array(wf.rois.get_concentrations('pancreas'))
    pancreas_tac -= pancreas_tac[0]

    input_tac = np.array(wf.rois.get_concentrations('aorta'))
    input_tac -= input_tac[0]
    input_tac_smoothed, input_tac_splines = express.spline_interpolation(input_tac, times, times_all)

    #generate parameters of R-curves
    pars = express.combine_pars(bv=np.arange(0.1, 0.5, 0.05),
                                a=np.arange(1, 15, 0.5),
                                loc=np.arange(0.1, 13, 0.3),
                                scale=np.arange(0.1, 6, 0.3))
    #generate possible TACs
    tacs = express.calc_tissue_tacs_mrx_3gamma(input_tac_smoothed,
                                               params=pars,
                                               time_steps=times_all,
                                               time_subset=times)
    #choose the best
    best_indices = getbest(tacs, pancreas_tac)
    #plot best fitted
    plt.subplot(211)
    plt.plot(times, pancreas_tac, 'o-g')
    plt.plot(times, tacs[:, best_indices], 'o-k', alpha=0.3)
    #plot pdf of R-curves
    times_for_gamma = np.arange(0, 100, 1)[..., None].repeat(len(best_indices), axis=1)
    pdfs = stats.gamma.pdf(times_for_gamma,
                           pars['a'][best_indices],
                           loc=pars['loc'][best_indices],
                           scale=pars['scale'][best_indices])
    print 'mtt', stats.gamma.mean(pars['a'][best_indices],
                                  loc=pars['loc'][best_indices],
                                  scale=pars['scale'][best_indices])
    print 'bv', pars['bv'][best_indices]
    plt.subplot(212)
    plt.plot(np.arange(0, 100, 1)[..., None], pdfs)
    plt.show()


