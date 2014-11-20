# -*- coding: utf8 -*-
import gc

import numpy as np
from scipy import interpolate, stats, signal
import matplotlib.pyplot as plt
import json

import pyximport
import itertools
pyximport.install()
import cssd

def get_tac(roi, vol4d, time):
    tac = []
    for t in range(len(time)):
        tac += np.median(vol4d[..., t][roi == 1]),
    return np.array(tac)


def run_trapz(input_conc, window):
    o = np.zeros(input_conc.shape)
    for i in range(len(input_conc)):
        fr = (i - window > 0) and i - window or 0
        o[i] = np.trapz(input_conc[fr:i]) / window
    return o


def spline_interpolation(conc, time, new_time, ss=0):
    tck = interpolate.splrep(time, conc, s=ss)
    return interpolate.splev(new_time, tck, der=0), tck


def calc_tissue_tac_mrx(input_tac, mtts, bvs, times, lags=(0,)):
    """
    Calculate Time/Attenuation Curve (TAC) of tissue from input TAC smoothed with spline


    Args:
      input_tac (tuple): is argument to scipy.interpolate.splint(..., tck, ...)
      mtts (iterable): mean transit time of tissue in seconds
      bvs (iterable): tissue blood volume. Should be between 0 and 1
      times (np.array): time steps of output TAC
      lag (iterable): time which input TAC needed to get to the tissue

    Returns:
      tacs(list): list of TACs
      params(list): list of tuples with TACs parameters (mtt,bv,lag)

    """

    if np.any(bvs < 0) or np.any(bvs > 1):
        raise ValueError('bvs should be in interval from 0 to 1')
    tacs = []
    params = []
    for im in mtts:
        for ib in bvs:
            for il in lags:
                tacs += calc_tissue_tac(input_tac, im, ib, np.array(times), il),
                params += (im, ib, il),
    # plt.plot(times,interpolate.splev(times, input_tac, der=0))
    # print times

    # plt.plot(times,np.array(tacs).T)
    # plt.show()
    return tacs, params



def calc_tissue_tac_mrx_conv(input_tac, time_steps, time_subset, rc_type, mtts, bvs, rc_sigma=(0.5,), lags=(0,)):
    """
    Calculate Time/Attenuation Curve (TAC) of tissue from input TAC smoothed with spline


    Args:
      input_tac (tuple): is argument to scipy.interpolate.splint(..., tck, ...)
      mtts (iterable): mean transit time of tissue in seconds
      bvs (iterable): tissue blood volume. Should be between 0 and 1
      times (np.array): time steps of output TAC
      lag (iterable): time which input TAC needed to get to the tissue

    Returns:
      tacs(list): list of TACs
      params(list): list of tuples with TACs parameters (mtt,bv,lag)

    """

    if np.any(bvs < 0) or np.any(bvs > 1):
        raise ValueError('bvs should be in interval from 0 to 1')

    tacs = []
    params = [(i_mtt, i_bv, i_lag, i_sigma) for i_mtt in mtts
              for i_bv in bvs
              for i_lag in lags
              for i_sigma in rc_sigma]
    for i in params:
        tacs.append(calc_tissue_tac_conv(input_tac,
                                         mtt=i[0],
                                         bv=i[1],
                                         time_steps=time_steps,
                                         t_subset=time_subset,
                                         rc_family=rc_type,
                                         sigma=i[3]))

    return tacs, params

def perf_pars_gen(bvs,mtts,sigmas,lags=(0,),bvs2=None,mtts2=None,sigmas2=None,lags2=None):
    if bvs2 is None:
        bvs2=bvs
    if mtts2 is None:
        mtts2 = mtts
    if sigmas2 is None:
        sigmas2=sigmas
    if lags2 is None:
        lags2=lags

    params11 = [(i_mtt, i_bv, i_lag, i_sigma, i2_mtt, i2_bv, i2_lag, i2_sigma) for i_mtt in mtts
                                                                                for i_bv in bvs[1:]
                                                                                for i_lag in lags
                                                                                for i_sigma in sigmas
                                                                                for i2_mtt in mtts2
                                                                                for i2_bv in bvs2[1:] if i_bv+i2_bv<=1 
                                                                                for i2_lag in lags2
                                                                                for i2_sigma in sigmas2]
    
    params00 = [(mtts[0], bvs[0], lags[0], sigmas[0], mtts2[0], bvs2[0], lags2[0], sigmas2[0]),]
    params01 = [(mtts[0], bvs[0], lags[0], sigmas[0], i2_mtt, i2_bv, i2_lag, i2_sigma) for i2_mtt in mtts2
                                                                                      for i2_bv in bvs2[1:]
                                                                                      for i2_lag in lags2
                                                                                      for i2_sigma in sigmas2]
    params10 = [(i_mtt, i_bv, i_lag, i_sigma, mtts2[0], bvs2[0], lags2[0], sigmas2[0]) for i_mtt in mtts
                                                                                for i_bv in bvs[1:]
                                                                                for i_lag in lags
                                                                                for i_sigma in sigmas]
    params = np.array(params11 + params00 + params01 + params10)
    print 'params shape', params.shape
    return params

def combine_pars1(**kwargs):
    lists_of_values = tuple([list(i) for i in kwargs.values()])
    print 'tuple generated',lists_of_values
    pars = np.array([i for i in itertools.product(*lists_of_values)])
    print 'array generated'
    out_dict = {}
    for k, vi in zip(kwargs.keys(), range(pars.shape[1])):
        out_dict[k] = pars[:, vi]
    return out_dict
def combine_pars(pars_to_combine,mtt_range=(0,np.inf),bv_range=(0,np.inf),bf_range=(0,np.inf),time_max=None,dist=None):
    lists_of_values = tuple([list(i) for i in pars_to_combine.values()])

    print 'tuple generated', lists_of_values
    combined_pars = itertools.product(*lists_of_values)
    filtered_pars=dict([(k,[]) for k in pars_to_combine.keys()])
    made_i=0
    saved_i=0
    while True:
        pars = np.array([i for i,ii in zip(combined_pars,range(2*10**6))])
        made_i+=len(pars)
        print 'made',made_i
        if len(pars)==0:
            break
        out_dict = {}
        for k, vi in zip(pars_to_combine.keys(), range(pars.shape[1])):
            out_dict[k] = pars[:, vi]
        fp = filter_pars_md(out_dict,mtt_range,bv_range,bf_range,time_max,dist)
        for k,v in fp.items():
            filtered_pars[k]=np.append(filtered_pars[k],(list(v)))

        saved_i+=len(fp.values()[0])
        print 'saved',saved_i
    return filtered_pars

def gen_fname(**kwargs):
    #bv,a,loc,scale,bf,mtt,full=True,prefix='pars',lag=0,
    full=True
    white_list=('prefix','dist','a','b','loc','scale','bv','lag','bf','mtt','maxtime')
    ranges = dict((k,kwargs[k]) for k in kwargs if k in white_list)
    #ranges={'bv':bv,'a':a,'loc':loc,'scale':scale,'bf':bf,'mtt':mtt,'prefix':prefix,'lag':lag}
    format_str = lambda x: format(x,'g')
    fname_l =[]
    for k in white_list:
        try:
            ranges[k]=map(format_str, ranges[k])
            s = '-'.join([k,]+ranges[k])
            fname_l.append(s)
        except:
            if k in ['prefix','dist']:
                fname_l.append(ranges[k])
            pass




    fname='_'.join(fname_l)

    """
    try:
        fname='{prefix}_a-{a[0]}-{a[1]}-{a[2]}_loc-{loc[0]}-{loc[1]}-{loc[2]}_scale-{scale[0]}-{scale[1]}-{scale[2]}_bv-{bv[0]}-{bv[1]}-{bv[2]}\
_lag-{lag}_bf-{bf[0]}-{bf[1]}_mtt-{mtt[0]}-{mtt[1]}'\
        .format(**ranges)
    except KeyError,s:
        print s
        fname='{prefix}_a-{a[0]}-{a[1]}-{a[2]}_loc-{loc[0]}-{loc[1]}-{loc[2]}_scale-{scale[0]}-{scale[1]}-{scale[2]}_bv-{bv[0]}-{bv[1]}-{bv[2]}\
_bf-{bf[0]}-{bf[1]}_mtt-{mtt[0]}-{mtt[1]}'\
        .format(**ranges)
    """
    return fname

def fname_to_range(fname):
    fname = fname.strip('.npy')
    out=[i.split('-') for i in fname.split('_')][1:]
    out=dict([[i[0],tuple(map(float,i[1:]))] for i in out])
    return out

def save_pars(pars,fname):
    pars = dict([[k,list(v)] for k,v in pars.items()])
    print 'presave made'
    with open(fname, 'wb') as fp:
        json.dump(pars, fp)
    print 'saved'
    return True
def load_pars(fname):
    with open(fname, 'rb') as fp:
        pars_subset = json.load(fp)
        pars_subset = dict([[k,np.array(v)] for k,v in pars_subset.items()])
    return pars_subset
def load_rc(fname):
    return np.load(fname)
def save_rc(rc,fname):
    np.save(fname,rc)

def filter_pars(pars,mtt_range=(0,np.inf),bv_range=(0,np.inf),bf_range=(0,np.inf),time_max=None,dist=None):
    mtt_array=(pars['a']*pars['scale']+pars['loc'])
    bv_array=pars['bv']
    bf_array=bv_array/mtt_array
    print 'all',pars['a'].shape
    mtt_mask = (mtt_range[0]<mtt_array) & (mtt_array<mtt_range[1])
    print 'mtt',np.sum(mtt_mask)
    bv_mask = (bv_range[0]<bv_array) & (bv_array<bv_range[1])
    print 'bv',np.sum(bv_mask)
    bf_mask = (bf_range[0]<bf_array) & (bf_array<bf_range[1])
    print 'bf', np.sum(bf_mask)
    mask = bf_mask & bv_mask & mtt_mask
    print np.sum(mask)
    return dict([[k,v[mask]] for k,v in pars.items()])

def filter_pars_md(pars,mtt_range=(0,np.inf),bv_range=(0,np.inf),bf_range=(0,np.inf),time_max=None,dist=None):
    print 'all',pars['a'].shape
    d=make_distribution(pars,dist)
    mtt_array=d.mean()
    mtt_mask = (mtt_range[0]<mtt_array) & (mtt_array<mtt_range[1])
    print 'mtt',np.sum(mtt_mask)
    pars = dict([[k,v[mtt_mask]] for k,v in pars.items()])
    d=make_distribution(pars,dist)
    mtt_array=mtt_array[mtt_mask]
    bv_array=pars['bv']
    bf_array=bv_array/mtt_array


    bv_mask = (bv_range[0]<bv_array) & (bv_array<bv_range[1])
    print 'bv',np.sum(bv_mask)
    bf_mask = (bf_range[0]<bf_array) & (bf_array<bf_range[1])
    print 'bf', np.sum(bf_mask)

    mask = bf_mask & bv_mask
    print np.sum(mask)
    del d


    if time_max and dist:
        pars = dict([[k,v[mask]] for k,v in pars.items()])
        d=make_distribution(pars,dist)
        interv=d.interval(0.99)
        interval_mask1=interv[1]<time_max
        interval_mask2=interv[0]>0
        interval_mask= interval_mask1 & interval_mask2
        print 'interval', np.sum(interval_mask)
        mask =interval_mask

    print np.sum(mask)
    return dict([[k,v[mask]] for k,v in pars.items()])



def calc_tissue_tac_mrx_conv2(input_tac, input_tac2, time_steps, time_subset, rc_type, mtts, bvs, rc_sigma=(0.5,), lags=(0,)):
    """
    Calculate Time/Attenuation Curve (TAC) of tissue from input TAC smoothed with spline


    Args:
      input_tac (tuple): is argument to scipy.interpolate.splint(..., tck, ...)
      mtts (iterable): mean transit time of tissue in seconds
      bvs (iterable): tissue blood volume. Should be between 0 and 1
      times (np.array): time steps of output TAC
      lag (iterable): time which input TAC needed to get to the tissue

    Returns:
      tacs(list): list of TACs
      params(list): list of tuples with TACs parameters (mtt,bv,lag)

    """

    if np.any(bvs < 0) or np.any(bvs > 1):
        raise ValueError('bvs should be in interval from 0 to 1')

    tacs = []

    params11 = [(i_mtt, i_bv, i_lag, i_sigma, i2_mtt, i2_bv, i2_lag, i2_sigma) for i_mtt in mtts
                                                                                for i_bv in bvs[1:]
                                                                                for i_lag in lags
                                                                                for i_sigma in rc_sigma
                                                                                for i2_mtt in mtts
                                                                                for i2_bv in bvs[1:] if i_bv+i2_bv<=1 
                                                                                for i2_lag in lags
                                                                                for i2_sigma in rc_sigma]
    
    params00 = [(mtts[0], bvs[0], lags[0], rc_sigma[0], mtts[0], bvs[0], lags[0], rc_sigma[0]),]
    params01 = [(mtts[0], bvs[0], lags[0], rc_sigma[0], i2_mtt, i2_bv, i2_lag, i2_sigma) for i2_mtt in mtts
                                                                                      for i2_bv in bvs[1:]
                                                                                      for i2_lag in lags
                                                                                      for i2_sigma in rc_sigma]
    params10 = [(i_mtt, i_bv, i_lag, i_sigma, mtts[0], bvs[0], lags[0], rc_sigma[0]) for i_mtt in mtts
                                                                                for i_bv in bvs[1:]
                                                                                for i_lag in lags
                                                                                for i_sigma in rc_sigma]
    params = params11 + params00 + params01 + params10

    print 'params calculated. There are', len(params), 'created'

    for i in params:

            tac1 = calc_tissue_tac_conv(input_tac,
                                                 mtt=i[0],
                                                 bv=i[1],
                                                 time_steps=time_steps,
                                                 t_subset=time_subset,
                                                 rc_family=rc_type,
                                                 sigma=i[3])
            tac2 = calc_tissue_tac_conv(input_tac2,
                                                 mtt=i[4],
                                                 bv=i[5],
                                                 time_steps=time_steps,
                                                 t_subset=time_subset,
                                                 rc_family=rc_type,
                                                 sigma=i[7])
            tacs.append(tac1+tac2)
            


    return tacs, params

def calc_tissue_tac_from_pars(input_tac, time_steps, params, time_subset=None, input_tac2=None):

    print 'params.shape',params.shape
    params1={'mtt':params[...,0],'bv':params[...,1],'sigma':params[...,3]}
    params2={'mtt':params[...,4],'bv':params[...,5],'sigma':params[...,7]}

    

    tac1 = calc_tissue_tacs_mrx(input_tac,
                                params = params1,
                                time_steps=time_steps,
                               t_subset=time_subset)
    if input_tac2:
        print 1
        tac2 = calc_tissue_tacs_mrx(input_tac2,
                                    params = params2,
                                    time_steps=time_steps,
                                    t_subset=time_subset)
    else:
       tac2=0
    tacs = tac1+tac2
    return tacs

def calc_tissue_tac(input_tac, mtt, bv, t, lag=0):
    """
    Calculate Time/Attenuation Curve (TAC) of tissue from input TAC smoothed with spline


    Args:
      input_tac (tuple): is argument to scipy.interpolate.splint(..., tck, ...)
      mtt (float): mean transit time of tissue in seconds
      bv (float): tissue blood volume. Should be between 0 and 1
      t (np.array): time steps of output TAC
      lag (float): time which input TAC needed to get to the tissue

    Returns:
      (np.array): tissue TAC with in defined time steps
    """
    if not 0 <= bv <= 1:
        raise ValueError('bv should be in interval from 0 to 1')
    if mtt == 0:
        mtt += 0.01

    t2 = t - lag
    t2[t2 < t[0]] = t[0]
    from_t = t2 - mtt
    from_t[from_t < t2[0]] = t2[0]
    final_arr = np.array([interpolate.splint(ft, tt, input_tac) for ft, tt in zip(from_t, t2)])
    return (final_arr * bv) / mtt

def calc_tissue_tac_conv(input_tac, mtt, bv, time_steps, t_subset=None, rc_family='trap', sigma=1):
    if rc_family == 'trap':
        rc_func = lambda x, y, z: make_rc_trap(x, y, z)
    elif rc_family == 'lognorm':
        rc_func = lambda x, y, z: make_rc_lognorm(x, y, z, sigma)

    rc = rc_func(mtt, bv, time_steps)
    out = np.convolve(input_tac, rc)

    if not t_subset is None:
        subset = [np.where(time_steps == i)[0][0] for i in t_subset]
        out = out[subset]
    return out

def calc_tissue_tacs_mrx(input_tac, params, time_steps, t_subset=None,t_res=1):
    """
    params - dictionary w keys: 'mtt','bv','sigma','lag'
    """
    from scipy import signal
    #print 'max params',np.max(params['mtt']),np.max(params['sigma']),np.max(params['bv'])
    out = np.array([[]])    
    #print t.shape
    params_num = params['mtt'].shape[0]

    for i_to,i_fr in zip(np.append(np.arange(50000,params_num,50000),params_num),
                                    np.arange(0,params_num,50000)):
        t = np.arange(0,np.max(time_steps),t_res)[...,None].repeat(len(params['mtt'][i_fr:i_to]),axis=1)
        rc = 1 - stats.gamma.cdf(t, 1, params['mtt'][i_fr:i_to], params['sigma'][i_fr:i_to])
        #add lags

        print 'rc shape', rc.shape
        rc /= np.sum(rc,axis=0)
        rc = params['bv'][i_fr:i_to] * rc 
        print 'rc max', np.max(rc)
        print 'zeros',params['mtt'][i_fr:i_to][np.where(np.sum(rc,axis=0)==0)],params['sigma'][i_fr:i_to][np.where(np.sum(rc,axis=0)==0)],params['bv'][i_fr:i_to][np.where(np.sum(rc,axis=0)==0)]
        #rc = s_rc / np.sum(rc,axis=0)
        #plt.plot(time_steps,input_tac)
        #plt.show()
        #plt.plot(t,rc)
        #plt.show()
    
        tacs = signal.fftconvolve(input_tac[:,None], rc)
        print 'tac shape', tacs.shape

        if not t_subset is None:
          subset = []
          sub0=0
          print sub0

          for i in  t_subset:
            print i,time_steps
            print np.where(time_steps == i-sub0)

            subset.append(np.where(time_steps == i-sub0)[0][0])

          try:
            out = np.append(out,tacs[subset],axis=1) 
          except ValueError,s:
            print s
            out = tacs[subset]
          print 'subset shape', out.shape
        else:
            out=tacs

    print out.shape, np.max(out)
    return out

def calc_rc_big(params, time_res, maxtime=100):
    params_num = params['a'].shape[0]
    out_rc = np.array([[]])
    for i_to,i_fr in zip(np.append(np.arange(50000,params_num,50000),params_num),
                                    np.arange(0,params_num,50000)):

        rc = calc_rc(dict([[k,v[i_fr:i_to]] for k,v in params.items()]),time_res,maxtime)
        try:
            out_rc = np.append(out_rc,rc,axis=1)
        except ValueError,s:
            print s
            out_rc = rc

    return out_rc

def calc_rc(params,time_res,maxtime=100):
    t = np.arange(0, maxtime, time_res)[...,None].repeat(len(params['a']),axis=1)
    rc = 1 - stats.gamma.cdf(t, params['a'], params['loc'], params['scale'])
    rc /= np.sum(rc,axis=0)
    return params['bv'] * rc

def make_distribution(params,dist):
    distribution={'gamma':{'func':stats.gamma,'white_list':('a','loc','scale')},
                  'beta':{'func':stats.beta,'white_list':('a','b','loc','scale')},
                  'alpha':stats.alpha,
                  'norm':stats.norm,
                  'lognorm':stats.lognorm}
    d=distribution[dist]
    return d['func'](**dict((k,params[k]) for k in params if k in d['white_list'] ))

def calc_pdfs(params, time_res, maxtime=100):
    params_num = params['a'].shape[0]
    out_rc = np.array([[]])
    for i_to,i_fr in zip(np.append(np.arange(50000,params_num,50000),params_num),
                                    np.arange(0,params_num,50000)):
        t = np.arange(0, maxtime, time_res)[...,None].repeat(len(params['a'][i_fr:i_to]),axis=1)
        rc = stats.gamma.pdf(t, params['a'][i_fr:i_to], params['loc'][i_fr:i_to], params['scale'][i_fr:i_to])

        try:
            out_rc = np.append(out_rc,rc,axis=1)
        except ValueError,s:
            print s
            out_rc = rc

    return out_rc

def calc_tac_gamma(input_tac, rc, time_subset=None, t_res=1, t_lag=0, loop_size=50000):
    """
    Compute tissue TACs from input TAC and tissue residue function with FFT convolution
    :param input_tac: 1d np.array of input TAC
    :param rc: nd np.array of residue curves
    :param time_subset: times in seconds that will be subseted from result TACS
    :param t_res: time resolution in seconds of input and residue TACs (should be same for the both TACs)
    :param t_lag: time in seconds needed for blood path from input to tissue
    :param loop_size: size of tacs calculated in one loop to prevent MemmoryError
    :return: nd np.array of subseted tissue TACs
    """
    rc_num = rc.shape[1]
    out = np.array([[]])
    time_steps=np.round(np.arange(time_subset[0], time_subset[-1] + 1, t_res),1)
    #lag input TAC
    input_tac= np.append(np.zeros(t_lag/t_res),input_tac)
    print rc_num
    for i_to,i_fr in zip(np.append(np.arange(loop_size,rc_num,loop_size),rc_num),
                                    np.arange(0,rc_num,loop_size)):
        print i_to,i_fr
        #print rc[:,i_fr:i_to].shape
        tacs_temp = signal.fftconvolve(input_tac[:,None], rc[:,i_fr:i_to])

        if not time_subset is None:
            subset = []
            sub0=0
            #print sub0

            for i in  time_subset:
                #print i,time_steps
                #print np.where(time_steps == i-sub0)

                subset.append(np.where(time_steps == i-sub0)[0][0])

            try:
                out = np.append(out,tacs_temp[subset],axis=1)
            except ValueError,s:
                print s
                out = tacs_temp[subset]
                #print 'subset shape', out.shape
        else:
            out=tacs_temp
    return out


def calc_tissue_tacs_mrx_3gamma(input_tac, params, time_steps, time_subset=None,t_res=1):
    """
    params - dictionary w keys: 'a','bv','loc','scale'
    """

    #print 'max params',np.max(params['mtt']),np.max(params['sigma']),np.max(params['bv'])
    out = np.array([[]])
    #print t.shape
    params_num = params['a'].shape[0]

    for i_to,i_fr in zip(np.append(np.arange(50000,params_num,50000),params_num),
                                    np.arange(0,params_num,50000)):
        t = np.arange(0,np.max(time_steps),t_res)[...,None].repeat(len(params['a'][i_fr:i_to]),axis=1)
        rc = 1 - stats.gamma.cdf(t, params['a'][i_fr:i_to], params['loc'][i_fr:i_to], params['scale'][i_fr:i_to])
        #add lags

        rc /= np.sum(rc,axis=0)
        rc = params['bv'][i_fr:i_to] * rc


        tacs = signal.fftconvolve(input_tac[:,None], rc)


        if not time_subset is None:
          subset = []
          sub0=0
          print sub0

          for i in  time_subset:
            print i,time_steps
            print np.where(time_steps == i-sub0)

            subset.append(np.where(time_steps == i-sub0)[0][0])

          try:
            out = np.append(out,tacs[subset],axis=1)
          except ValueError,s:
            print s
            out = tacs[subset]
          print 'subset shape', out.shape
        else:
            out=tacs

    print out.shape, np.max(out)
    return out


def make_rc_trap(mtt, bv, ts):
    ts_fr0 = np.arange(0, ts.max(), ts[1] - ts[0])
    rc = np.zeros(ts_fr0.shape[0])
    rc[ts_fr0 < mtt] = 1
    print np.sum(rc)
    return bv * rc / mtt


def make_rc_lognorm(mtt, bv, ts, sigma=0.01):
    ts_fr0 = np.arange(0, ts.max(), ts[1] - ts[0])

    rc = 1 - stats.lognorm.cdf(ts_fr0, sigma, 0, mtt)
    rc /= np.sum(rc)
    s_rc = bv * rc
    # print rc
    return s_rc


def make_rc_gamma(mtt, bv, ts, sigma=0.01):
    ts_fr0 = np.arange(0, ts.max(), ts[1] - ts[0])
    rc = 1 - stats.gamma.cdf(ts_fr0, 1, mtt, sigma)
    s_rc = bv * rc
    return s_rc / np.sum(rc)


def calculate_mtt_bv(tac, example_mrx, mtt_vector, bv_vector):
    # tac=np.array(tac)
    # print tac.shape,example_mrx.shape
    diff = (example_mrx - tac[None, :, None])
    ssd = np.sum(diff * diff, axis=1)
    mtt_idx, bv_idx = np.where(ssd == np.min(ssd))
    return mtt_vector[mtt_idx], bv_vector[bv_idx]


def make_map(vol4d, input_tac, mtt_range, bv_range, times, lag_range):
    mtt_array = np.arange(mtt_range[0], mtt_range[1], mtt_range[2])
    bv_array = np.arange(bv_range[0], bv_range[1], bv_range[2])
    lag_array = np.arange(lag_range[0], lag_range[1], lag_range[2])

    reference_tacs, reference_pars = calc_tissue_tac_mrx(input_tac, mtt_array, bv_array, times, lag_array)
    reference_tacs = np.array(reference_tacs)
    vol4d_2d = vol4d.reshape((np.prod(vol4d.shape[:-1]), vol4d.shape[-1]))

    perfusion = np.zeros((vol4d_2d.shape[0], 3))

    done = 0.
    loops = reference_tacs.shape[0]
    print loops
    for ref_tac, ref_pars in zip(reference_tacs, reference_pars):
        diff = vol4d_2d - ref_tac
        done += 1
        progress_bar = round(done / loops)
        if progress_bar > 0 and progress_bar % 5 == 0:
            print done / loops
        ssd2 = np.sum(diff * diff, axis=1)
        try:
            mask_array = ssd1 > ssd2
        except NameError:
            mask_array = np.ones(ssd2.shape, dtype=np.bool)
            ssd1 = np.copy(ssd2)
        perfusion[mask_array] = ref_pars
        ssd1[mask_array] = ssd2[mask_array]

    bv_vol = perfusion[:, 1].reshape(vol4d.shape[:-1]) * 100
    mtt_vol = perfusion[:, 0].reshape(vol4d.shape[:-1])
    lag_vol = perfusion[:, 2].reshape(vol4d.shape[:-1])
    ssd_vol = np.array(ssd1).reshape(vol4d.shape[:-1])
    bf_vol = bv_vol / (mtt_vol / 60.)
    return bv_vol, mtt_vol, bf_vol, lag_vol, ssd_vol


def make_map2(vol4d, input_tac, mtt_range, bv_range, times, lag_range):
    mtt_array = np.arange(mtt_range[0], mtt_range[1], mtt_range[2])
    bv_array = np.arange(bv_range[0], bv_range[1], bv_range[2])
    lag_array = np.arange(lag_range[0], lag_range[1], lag_range[2])
    print times
    reference_tacs, reference_pars = calc_tissue_tac_mrx(input_tac, mtt_array, bv_array, times, lag_array)
    reference_tacs = np.array(reference_tacs)
    vol4d_2d = vol4d.reshape((np.prod(vol4d.shape[:-1]), vol4d.shape[-1]))

    perfusion = []
    ssds = []
    loops = np.prod(vol4d.shape[:-1])
    done = 0.
    print reference_tacs.shape, vol4d.shape,
    for real_tac in vol4d_2d:
        done += 1
        if done % 50000 == 0:
            print done / loops
        diff = reference_tacs - real_tac
        ssd = np.sum(diff * diff, axis=-1)

        # print np.where(ssd == min_ssd)[0][0]
        min_idx = ssd.argmin(axis=0)
        m = reference_pars[min_idx]
        perfusion.append(m)
        ssds.append(ssd[min_idx])

    perfusion = np.array(perfusion)
    bv_vol = perfusion[:, 1].reshape(vol4d.shape[:-1]) * 100
    mtt_vol = perfusion[:, 0].reshape(vol4d.shape[:-1])
    lag_vol = perfusion[:, 2].reshape(vol4d.shape[:-1])
    ssd_vol = np.array(ssds).reshape(vol4d.shape[:-1])
    bf_vol = bv_vol / (mtt_vol / 60.)
    return bv_vol, mtt_vol, bf_vol, lag_vol, ssd_vol


def make_map_conv(vol4d, times, input_tac, mtt_range, bv_range, lag_range, sigma_range, rc_type='lognorm', time_res=1):
    mtt_array = np.arange(mtt_range[0], mtt_range[1], mtt_range[2])
    bv_array = np.arange(bv_range[0], bv_range[1], bv_range[2])
    lag_array = np.arange(lag_range[0], lag_range[1], lag_range[2])
    sigma_array = np.arange(sigma_range[0], sigma_range[1], sigma_range[2])

    print times
    print 'mtt', mtt_array
    print 'bv', bv_array
    print 'lag', lag_array
    print 'sigma', sigma_array
    new_time_steps = np.arange(times[0], times[-1] + 1, time_res)
    input_tac_smoothed, input_tac_splines = spline_interpolation(input_tac, times, new_time_steps)
    input_tac_smoothed[new_time_steps < times[0]] = input_tac[0]

    reference_tacs, reference_pars = calc_tissue_tac_mrx_conv(input_tac_smoothed,
                                                              time_steps=new_time_steps,
                                                              time_subset=times,
                                                              rc_type=rc_type,
                                                              mtts=mtt_array,
                                                              bvs=bv_array,
                                                              rc_sigma=sigma_array,
                                                              lags=lag_array)
    reference_tacs = np.array(reference_tacs)
    # plt.plot(times, reference_tacs.T, alpha=0.1)
    #plt.plot(times, input_tac)
    #plt.plot(new_time_steps, input_tac_smoothed)
    #plt.show()
    vol4d_2d = vol4d.reshape((np.prod(vol4d.shape[:-1]), vol4d.shape[-1]))

    perfusion = []
    ssds = []
    loops = np.prod(vol4d.shape[:-1])
    done = 0.
    print reference_tacs.shape, vol4d.shape
    for real_tac in vol4d_2d:
        done += 1
        diff = reference_tacs - real_tac
        ssd = np.sum(diff * diff, axis=-1)
        min_idx = ssd.argmin()
        m = reference_pars[min_idx]
        perfusion.append(m)
        ssds.append(ssd[min_idx])
        if done % 50000 == 0:
            print done / loops, ssd[min_idx], m, ssd.shape
            """
            subset = ssd.argsort()[:100]
            plt.subplot(211)
            plt.plot(times,input_tac)
            plt.plot(times,real_tac)
            plt.plot(times,reference_tacs[subset].T,alpha=0.1)
            plt.subplot(212)
            plt.plot(np.array(reference_pars)[:,3][subset],ssd[subset],'o')
            plt.show()
            """
            # print np.where(ssd == min_ssd)[0][0]

    perfusion = np.array(perfusion)
    bv_vol = perfusion[:, 1].reshape(vol4d.shape[:-1]) * 100
    mtt_vol = perfusion[:, 0].reshape(vol4d.shape[:-1])
    lag_vol = perfusion[:, 2].reshape(vol4d.shape[:-1])
    sigma_vol = perfusion[:, 3].reshape(vol4d.shape[:-1])
    ssd_vol = np.array(ssds).reshape(vol4d.shape[:-1])
    bf_vol = bv_vol / (mtt_vol / 60.)
    return bv_vol, mtt_vol, bf_vol, sigma_vol, lag_vol, ssd_vol


def make_map_conv_cython(vol4d, times, input_tac, mtt_range, bv_range, lag_range, sigma_range, rc_type='lognorm',
                         time_res=1, input_tac2=None):
    mtt_array = np.arange(mtt_range[0], mtt_range[1], mtt_range[2])
    bv_array = np.arange(bv_range[0], bv_range[1], bv_range[2])
    lag_array = np.arange(lag_range[0], lag_range[1], lag_range[2])
    sigma_array = np.arange(sigma_range[0], sigma_range[1], sigma_range[2])

    print times
    print 'mtt', mtt_array
    print 'bv', bv_array
    print 'lag', lag_array
    print 'sigma', sigma_array
    new_time_steps = np.arange(times[0], times[-1] + 1, time_res)
    input_tac_smoothed, input_tac_splines = spline_interpolation(input_tac, times, new_time_steps)
    input_tac_smoothed[new_time_steps < times[0]] = input_tac[0]
    if input_tac2 is None:

        reference_tacs, reference_pars = calc_tissue_tac_mrx_conv(input_tac_smoothed,
                                                                  time_steps=new_time_steps,
                                                                  time_subset=times,
                                                                  rc_type=rc_type,
                                                                  mtts=mtt_array,
                                                                  bvs=bv_array,
                                                                  rc_sigma=sigma_array,
                                                                  lags=lag_array)
    else:
        print 'double input'
        input_tac_smoothed2, input_tac_splines2 = spline_interpolation(input_tac2, times, new_time_steps)
        input_tac_smoothed2[new_time_steps < times[0]] = input_tac2[0]

        reference_tacs, reference_pars = calc_tissue_tac_mrx_conv2(input_tac_smoothed,
                                                                    input_tac_smoothed2,
                                                                    time_steps=new_time_steps,
                                                                    time_subset=times,
                                                                    rc_type=rc_type,
                                                                    mtts=mtt_array,
                                                                    bvs=bv_array,
                                                                    rc_sigma=sigma_array,
                                                                    lags=lag_array)
    print 'Tacs created!'
    reference_tacs = np.array(reference_tacs)



    vol4d_2d = vol4d.reshape((np.prod(vol4d.shape[:-1]), vol4d.shape[-1]))

    print 'references', vol4d_2d.dtype, reference_tacs.dtype

    pars_indexes, ssds = np.array(cssd.ssd_int(vol4d_2d, reference_tacs.astype("float32")), dtype=int)
    print pars_indexes.shape
    perfusion = np.array(reference_pars)[pars_indexes]

    print 1
    bv_vol = perfusion[:, 1].reshape(vol4d.shape[:-1]) * 100
    mtt_vol = perfusion[:, 0].reshape(vol4d.shape[:-1])
    lag_vol = perfusion[:, 2].reshape(vol4d.shape[:-1])
    sigma_vol = perfusion[:, 3].reshape(vol4d.shape[:-1])

    print 2
    ssd_vol = np.array(ssds, dtype=float).reshape(vol4d.shape[:-1])
    print 3
    bf_vol = bv_vol / (mtt_vol / 60.)
    portal_volumes={}
    if not input_tac2 is None:

        portal_volumes['bv'] = perfusion[:, 5].reshape(vol4d.shape[:-1]) * 100
        portal_volumes['mtt'] = perfusion[:, 4].reshape(vol4d.shape[:-1])
        portal_volumes['lag'] = perfusion[:, 6].reshape(vol4d.shape[:-1])
        portal_volumes['sigma'] = perfusion[:, 7].reshape(vol4d.shape[:-1])
        portal_volumes['bf'] = portal_volumes['bv'] / (portal_volumes['mtt'] / 60.)


    return bv_vol, mtt_vol, bf_vol, sigma_vol, lag_vol, ssd_vol, portal_volumes

def make_map_conv_cython2(vol4d, times, input_tac, mtt_range, bv_range, lag_range, sigma_range, rc_type='lognorm',
                         time_res=1, input_tac2=None, vol4d_mask=None ):
    mtt_array = np.arange(mtt_range[0], mtt_range[1], mtt_range[2])
    bv_array = np.arange(bv_range[0], bv_range[1], bv_range[2])
    lag_array = np.arange(lag_range[0], lag_range[1], lag_range[2])
    sigma_array = np.arange(sigma_range[0], sigma_range[1], sigma_range[2])

    print times
    print 'mtt', mtt_array
    print 'bv', bv_array
    print 'lag', lag_array
    print 'sigma', sigma_array
    new_time_steps = np.arange(times[0], times[-1] + 1, time_res)
    input_tac_smoothed, input_tac_splines = spline_interpolation(input_tac, times, new_time_steps)
    input_tac_smoothed[new_time_steps < times[0]] = input_tac[0]
    if input_tac2 is None:

        reference_tacs, reference_pars = calc_tissue_tac_mrx_conv(input_tac_smoothed,
                                                                  time_steps=new_time_steps,
                                                                  time_subset=times,
                                                                  rc_type=rc_type,
                                                                  mtts=mtt_array,
                                                                  bvs=bv_array,
                                                                  rc_sigma=sigma_array,
                                                                  lags=lag_array)
    else:
        print 'double input'
        input_tac_smoothed2, input_tac_splines2 = spline_interpolation(input_tac2, times, new_time_steps)
        input_tac_smoothed2[new_time_steps < times[0]] = input_tac2[0]
        """
        plt.plot(times,input_tac2)
        plt.plot(new_time_steps,input_tac_smoothed2)
        plt.plot(times,input_tac)
        plt.plot(new_time_steps,input_tac_smoothed)
        plt.show()
        """


        reference_tacs, reference_pars = calc_tissue_tac_mrx_conv2(input_tac_smoothed,
                                                                    input_tac_smoothed2,
                                                                    time_steps=new_time_steps,
                                                                    time_subset=times,
                                                                    rc_type=rc_type,
                                                                    mtts=mtt_array,
                                                                    bvs=bv_array,
                                                                    rc_sigma=sigma_array,
                                                                    lags=lag_array)
    print 'Tacs created!'
    reference_tacs = np.array(reference_tacs)
    print reference_pars
    #plt.plot(times, reference_tacs.T)
    #plt.show()

    if type(vol4d_mask)==np.ndarray:
        print vol4d_mask.shape,vol4d.shape
        vol4d_shape = vol4d.shape[:-1]

        mask_row_num = vol4d_mask[vol4d_mask==1].size
        vol4d = vol4d[vol4d_mask==1,:]
        print vol4d.shape
        vol4d_2d = np.array(vol4d,order='C')
        #.reshape((mask_row_num, vol4d.shape[-1]),order='C')
        print
    else:
        vol4d_2d = vol4d.reshape((np.prod(vol4d.shape[:-1]), vol4d.shape[-1]))

    print 'references', vol4d_2d.dtype, reference_tacs.dtype,vol4d_2d.shape

    pars_indexes, ssds = np.array(cssd.ssd_int(vol4d_2d, reference_tacs.astype("float32")), dtype=int)
    print pars_indexes
    print pars_indexes.shape,np.array(reference_pars).shape
    perfusion = np.array(reference_pars)[pars_indexes]
    print perfusion.shape,np.max(perfusion,axis=0),perfusion.dtype,np.max(perfusion[:,1])
    print dir()
    if type(vol4d_mask)==np.ndarray:

        bv_vol,mtt_vol,lag_vol,sigma_vol,ssd_vol = [np.zeros(vol4d_shape) for i in range(5)]

        print vol4d_mask.shape,bv_vol.shape,perfusion[:, 1].shape
        print len(bv_vol[vol4d_mask==1]),len(perfusion[:, 1])
        bv_vol[vol4d_mask==1] = perfusion[:, 1] * 100

        mtt_vol[vol4d_mask==1] = perfusion[:, 0]
        lag_vol[vol4d_mask==1] = perfusion[:, 2]
        sigma_vol[vol4d_mask==1] = perfusion[:, 3]
        bf_vol = bv_vol / ((mtt_vol+1) / 60.)
        ssd_vol[vol4d_mask==1] = np.array(ssds, dtype=float)
        print np.max(bv_vol),np.max(mtt_vol),np.max(lag_vol),np.max(sigma_vol),np.max(bf_vol),np.max(ssd_vol)
        portal_volumes=dict([[i,np.zeros(vol4d_shape)] for i in ['bv','mtt','lag','sigma','bf']])
        print 1
        portal_volumes['bv'][vol4d_mask==1] = perfusion[:, 5] * 100
        portal_volumes['mtt'][vol4d_mask==1] = perfusion[:, 4]
        portal_volumes['lag'][vol4d_mask==1] = perfusion[:, 6]
        portal_volumes['sigma'][vol4d_mask==1] = perfusion[:, 7]
        portal_volumes['bf'] = portal_volumes['bv'] / ((1+portal_volumes['mtt']) / 60.)
        print 1
        return bv_vol, mtt_vol, bf_vol, sigma_vol, lag_vol, ssd_vol, portal_volumes


    print 1
    bv_vol = perfusion[:, 1].reshape(vol4d.shape[:-1]) * 100
    mtt_vol = perfusion[:, 0].reshape(vol4d.shape[:-1])
    lag_vol = perfusion[:, 2].reshape(vol4d.shape[:-1])
    sigma_vol = perfusion[:, 3].reshape(vol4d.shape[:-1])

    print 2
    ssd_vol = np.array(ssds, dtype=float).reshape(vol4d.shape[:-1])
    print 3
    bf_vol = bv_vol / (mtt_vol / 60.)
    portal_volumes={}
    if not input_tac2 is None:

        portal_volumes['bv'] = perfusion[:, 5].reshape(vol4d.shape[:-1]) * 100
        portal_volumes['mtt'] = perfusion[:, 4].reshape(vol4d.shape[:-1])
        portal_volumes['lag'] = perfusion[:, 6].reshape(vol4d.shape[:-1])
        portal_volumes['sigma'] = perfusion[:, 7].reshape(vol4d.shape[:-1])
        portal_volumes['bf'] = portal_volumes['bv'] / (portal_volumes['mtt'] / 60.)


    return bv_vol, mtt_vol, bf_vol, sigma_vol, lag_vol, ssd_vol, portal_volumes


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    dict3 = combine_pars(a=np.arange(10),
                         b=np.arange(10,50,10),
                         c=np.arange(99,999,121),)

    print dict3





