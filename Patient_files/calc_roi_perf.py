__author__ = 'denest'
import sys
import os

sys.path.insert(0, '/home/denest/')
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from PERFetc2 import workflow
from PERFetc2.perfusion import express


def getbest(tacs, reference, best_num=10):
    diff = tacs - reference[:, None]
    diff *= diff
    diff = np.sum(diff, axis=0)
    ssd_threshold = np.sort(diff)[best_num]
    print 'ssd',np.sort(diff)[:best_num]
    return np.where(diff < ssd_threshold)[0]

def generate_tacs(tacs_folder,rc_folder,pars_folder,lags=(0,)):
    for p, d, f in os.walk(rc_folder):
        print p
        pars = {}
        tacs = np.array([[]])
        for i in f:
            rc_subset = express.load_rc(os.path.join(p, i))
            ranges = express.fname_to_range(i)
            pars_name = express.gen_fname(prefix='pars',
                                              full=True,
                                              **ranges)
            pars_subset = express.load_pars(os.path.join(pars_folder, pars_name))
            for l in lags:
                print i


                tacs_gen = express.calc_tac_gamma(input_tac_smoothed, rc_subset, time_subset=times, t_res=TIME_RESOLUTION, t_lag=1,
                                                  loop_size=50000)

                #tacs = express.append_result(tacs, tacs_gen)
                """
                for k in pars_subset:
                    try:
                        pars[k] = np.append(pars[k],pars_subset[k])
                    except KeyError, s:
                        print s
                        pars[k] = pars_subset[k]
                """
                ranges['lag']=l
                tacs_name = express.gen_fname(prefix='tacs',
                                              full=True,
                                              **ranges)
                print tacs_name
                print ranges
                express.save_rc(tacs_gen, os.path.join(tacs_folder, tacs_name))

        print tacs.shape,pars.values()[0].shape

    return tacs,pars



# choose_tacs

if __name__ == "__main__":
    ROOT_FOLDER_LIN = '/media/WORK___/_PERF/BENDER  ^V.A/20111206_1445'
    FILES_FOLDER = '/home/denest/PERFetc_files'
    PARS_FOLDER = 'pars'
    RC_FOLDER = 'rc'
    root_folder = os.path.abspath(FILES_FOLDER)
    pars_folder = os.path.join(root_folder, PARS_FOLDER)
    rc_folder = os.path.join(root_folder, RC_FOLDER)
    tacs_folder = os.path.join(ROOT_FOLDER_LIN,'tacs')
    if not os.path.isdir(tacs_folder):
        os.mkdir(tacs_folder)

    TIME_RESOLUTION = 0.5

    # setup patient rois
    wf = workflow.Workflow(ROOT_FOLDER_LIN)
    wf.update_label()
    #get time steps
    times = wf.rois.get_time_list()
    times_all = np.arange(times[0], times[-1] + 1, TIME_RESOLUTION)
    #get TACs
    pancreas_tac = np.array(wf.rois.get_concentrations('pancreas'))
    pancreas_tac -= pancreas_tac[0]

    input_tac = np.array(wf.rois.get_concentrations('aorta'))
    input_tac -= input_tac[0]
    input_tac_smoothed, input_tac_splines = express.spline_interpolation(input_tac, times, times_all)
    plt.plot(times,input_tac)
    plt.plot(times,pancreas_tac)
    plt.plot(times_all,input_tac_smoothed)
    plt.show()
    #generate parameters of R-curves

    #generate possible TACs
    #generate_tacs(tacs_folder,rc_folder,pars_folder,lags=(0,1,2,3,4,5,6))
    for p, d, f in os.walk(tacs_folder):
        print p
        pars = {}
        tacs = np.array([[]])
        for i in f:
            ranges = express.fname_to_range(i)
            if ranges['mtt'][0]>30:
                continue
            pars_name = express.gen_fname(prefix='pars',
                                              full=True,
                                              **ranges)
            pars_subset = express.load_pars(os.path.join(pars_folder, pars_name))

            print i
            tacs_name = express.gen_fname(prefix='tacs',
                                          full=True,
                                          **ranges)

            tacs_gen = express.load_rc(os.path.join(tacs_folder, i))
            pars_subset['lag']=np.repeat(ranges['lag'],pars_subset['a'].shape)
            tacs = express.append_result(tacs, tacs_gen)
            print tacs.shape
            for k in pars_subset:
                try:
                    pars[k] = np.append(pars[k],pars_subset[k])
                except KeyError, s:
                    print s
                    pars[k] = pars_subset[k]


                print tacs_name
                print ranges


    #choose the best
    best_indices = getbest(tacs, pancreas_tac,50)
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
    print pars['a'][best_indices], pars['loc'][best_indices], pars['scale'][best_indices],pars['lag'][best_indices]
    print 'mtt', stats.gamma.mean(pars['a'][best_indices],
                                  loc=pars['loc'][best_indices],
                                  scale=pars['scale'][best_indices])
    print 'bv', pars['bv'][best_indices]

    plt.subplot(212)
    plt.plot(np.arange(0, 100, 1)[..., None], pdfs)
    plt.show()


