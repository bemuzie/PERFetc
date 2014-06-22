# -*- coding: utf8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate, stats
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
        tacs += calc_tissue_tac_conv(input_tac,
                                     mtt=i[0],
                                     bv=i[1],
                                     time_steps=time_steps,
                                     t_subset=time_subset,
                                     rc_family=rc_type,
                                     sigma=i[3]),

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


def make_rc_trap(mtt, bv, ts):
    ts_fr0 = np.arange(0, ts.max(), ts[1] - ts[0])
    rc = np.zeros(ts_fr0.shape[0])
    rc[ts_fr0 < mtt] = 1
    print np.sum(rc)
    return bv * rc / mtt


def make_rc_lognorm(mtt, bv, ts, sigma=0.01):
    ts_fr0 = np.arange(0, ts.max(), ts[1] - ts[0])

    rc = 1 - stats.lognorm.cdf(ts_fr0, sigma, 0, mtt)
    s_rc = bv * rc
    # print rc
    return s_rc / np.sum(rc)


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
    plt.plot(times, reference_tacs.T, alpha=0.1)
    plt.plot(times, input_tac)
    plt.plot(new_time_steps, input_tac_smoothed)
    plt.show()
    vol4d_2d = vol4d.reshape((np.prod(vol4d.shape[:-1]), vol4d.shape[-1]))

    perfusion = []
    ssds = []
    loops = np.prod(vol4d.shape[:-1])
    done = 0.
    print reference_tacs.shape, vol4d.shape
    for real_tac in vol4d_2d:
        done += 1
        if real_tac.max() == real_tac[0]:
            perfusion.append((100, 0, 0, 0))
            ssds.append(100000)
            continue

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

    simple_tac = calc_tissue_tac(input_tac_pars, 20, 0.4, t)
    simple_tac2 = calc_tissue_tac_conv(input_tac_s, 20, 0.4, new_times, t, 'trap')
    simple_tac3 = calc_tissue_tac_conv(input_tac_s, 20, 0.4, new_times, t, 'lognorm', 0.3)

    plt.subplot(2, 1, 1)
    # plt.plot(t, a)
    plt.plot(t, simple_tac, 'r')
    plt.plot(t, simple_tac2, 'b')
    plt.plot(t, simple_tac3, 'g')
    plt.subplot(2, 1, 2)
    plt.plot(make_rc_lognorm(20, 0.4, new_times, 0.3), 'r')
    plt.plot(make_rc_trap(20, 0.4, new_times), 'g')
    plt.show()


