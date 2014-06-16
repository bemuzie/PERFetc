# -*- coding: utf8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.integrate import quadrature
import nibabel as nib
import matplotlib.pyplot as plt

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
    print bvs < 0
    print bvs > 1
    print bvs
    if np.any(bvs < 0) or np.any(bvs > 1):
        raise ValueError('bvs should be in interval from 0 to 1')
    print times

    print times
    tacs = []
    params = []
    for im in mtts:
        for ib in bvs:
            for il in lags:
                tacs += calc_tissue_tac(input_tac, im, ib, np.array(times), il),
                params += (im, ib, il),
    #plt.plot(times,interpolate.splev(times, input_tac, der=0))
    #print times

    #plt.plot(times,np.array(tacs).T)
    #plt.show()
    return tacs, params


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
        mtt+=0.01

    t2 = t-lag
    t2[t2<t[0]] = t[0]
    from_t = t2 - mtt
    from_t[from_t < t2[0]] = t2[0]
    final_arr = np.array([interpolate.splint(ft, tt, input_tac) for ft, tt in zip(from_t, t2)])
    return (final_arr * bv) / mtt


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

    perfusion = np.zeros((vol4d_2d.shape[0],3))

    done = 0.
    loops = reference_tacs.shape[0]
    print loops
    for ref_tac,ref_pars in zip(reference_tacs,reference_pars):
        diff = vol4d_2d - ref_tac
        done += 1
        progress_bar = round(done/loops)
        if progress_bar>0 and progress_bar%5 == 0:
            print done/loops
        ssd2 = np.sum(diff*diff, axis=1)
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
    bf_vol = (bv_vol) / (mtt_vol / 60.)
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
    print reference_tacs.shape,vol4d.shape,
    for real_tac in vol4d_2d:
        done += 1
        if done%50000 == 0:
            print done/loops
        diff = reference_tacs - real_tac
        ssd = np.sum(diff * diff, axis=-1)

        #print np.where(ssd == min_ssd)[0][0]
        min_idx = ssd.argmin(axis=0)
        m = reference_pars[min_idx]
        perfusion.append(m)
        ssds.append(ssd[min_idx])

    perfusion = np.array(perfusion)
    bv_vol = perfusion[:, 1].reshape(vol4d.shape[:-1]) * 100
    mtt_vol = perfusion[:, 0].reshape(vol4d.shape[:-1])
    lag_vol = perfusion[:, 2].reshape(vol4d.shape[:-1])
    ssd_vol = np.array(ssds).reshape(vol4d.shape[:-1])
    bf_vol = (bv_vol) / (mtt_vol / 60.)
    return bv_vol, mtt_vol, bf_vol, lag_vol, ssd_vol


if __name__ == "__main__":
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

    t = [10,
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

    print len(t)

    # plt.plot(run_sum(a,1,10))
    a = np.array(a)
    a -= a[0]
    t = np.array(t)

    p = interp1d(t, a, kind='cubic')
    # plt.plot(np.arange(np.min(t),np.max(t),.1),p(np.arange(np.min(t),np.max(t),.1)),color='k')
    # print quadrature(p,[1,2,3],[2,3,4])
    # plt.plot(t,run_trapz(a,1,4),color='red')
    spl, s = spline_interpolation(a, t, np.arange(8, 90, .1))

    mtts = np.arange(5, 40, 1)
    bvs = np.arange(0.1, 1, 0.1)

    pancreas = calc_tissue_tac_mrx(s, mtts, bvs, t)

    """
    curves_mrx = np.zeros([len(a),len(range(10))])
    print curves_mrx.shape
    for j in range(10):
        curves_mrx[:,j]=run_trapz(a,1,j)
    plt.plot(curves_mrx,color='red')
    plt.plot(curves_mrx*.2,color='blue')

    """

    plt.plot(t, a)

    path_to4d = u'/home/denest/PERF_volumes/MOSKOVTSEV  V.I. 18.01.1938/20140610_632/NII/FILTERED_TOSHIBA_REG/4D00.nii'
    nii4 = nib.load(path_to4d)
    niiData = nii4.get_data()
    # bv_vol=np.zeros(niiData.shape[:-1])
    # mtt_vol=np.zeros(niiData.shape[:-1])
    # out_vol = np.zeros(niiData.shape[:-1]+(3,))
    print niiData[..., 0][..., None].shape, niiData.shape

    # niiData=np.concatenate((niiData[...,0][...,None],niiData),axis=3)

    """
    for x in range(niiData.shape[0]):
        for y in range(niiData.shape[1]):
            for z in range(niiData.shape[2]):
                #print niiData[x,y,z].shape
                out_vol[x,y,z,:-1] = calculate_mtt_bv(niiData[x,y,z],pancreas,mtts,bvs)
    out_vol[...,-1] = (out_vol[...,1]*100.)/(out_vol[...,0]/60.)
    """
    niiData = niiData - niiData[..., 0][..., None]
    out_vol = make_map(niiData, s, mtt_range=(1, 80, 2), bv_range=(0.1, 1, 0.1), t=t, lag_range=(0, 5, 1))

    nii_im = nib.Nifti1Image(out_vol, nii4.get_header().get_sform(), nii4.get_header())

    nib.nifti1.save(nii_im,
                    u'/home/denest/PERF_volumes/MOSKOVTSEV  V.I. 18.01.1938/20140610_632/NII/FILTERED_TOSHIBA_REG/4D00_bf_map.nii')

    plt.show()

