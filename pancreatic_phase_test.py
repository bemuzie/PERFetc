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

print pancreatic
tacs=[]
bad_tacs=[]
maximums = np.zeros(t.shape[0])
for mtt in np.arange(8, 30,2):
    for bv in np.arange(0.3, 0.8, 0.2):
        for s in np.arange(0.01,1.5,0.2):
            tum_tac = calc_tissue_tac_conv(input_tac_s, mtt, bv, new_times, t, 'lognorm', s)
            tum_tac/=np.max(tum_tac)
            maximums[np.where(tum_tac==np.max(tum_tac))]+=1

            tacs.append(tum_tac)
            plt.subplot(311)
            if tum_tac[7] != np.max(tum_tac):
                bad_tacs.append(tum_tac)
                plt.plot(t,tum_tac,'o-',color=(mtt/30.,bv,0.8))

            #diff = simple_tac3-tum_tac
            """
            if diff[pancreatic] < 10 and diff[pancreatic] > -10 and np.any(diff[:pancreatic]>10) and np.all(diff[pancreatic:]<10) and np.all(diff[pancreatic:]>-10):
                print diff
                print diff[pancreatic]
                print mtt,bv,s



                simple_tac4 = calc_tissue_tac_conv(input_tac_s, mtt, bv, new_times, t, 'lognorm', s)

                #plt.subplot(2, 1, 1)
                #plt.plot(t, a/10.)
                #plt.plot(t, simple_tac, 'r')
                plt.plot(t, simple_tac3, 'o-g')
                plt.fill_between(t, simple_tac3-10,simple_tac3+10,color='green',alpha=0.5)
                for i in t:
                    plt.axvline(i, ls='--')
                #plt.plot(t, tum_tac, 'o-b')
                #plt.plot(t, simple_tac3-tum_tac, 'o-k')

                #plt.plot(t, simple_tac2, 'o-b')
                plt.plot(t, simple_tac4, 'o-r')

                #plt.plot(t, simple_tac4, 'o--g')
                plt.plot(t, simple_tac3-simple_tac4, 'o-k')
                #plt.plot(t, simple_tac3-simple_tac4, 'o--k')
                plt.axhline(10)
                #plt.subplot(2, 1, 2)
                #plt.plot(make_rc_lognorm(20, 0.2, new_times, 0.6), 'r')
                #plt.plot(make_rc_trap(20, 0.4, new_times), 'g')

                plt.savefig('2mtt%s_bv%s_s%s.png'%(mtt,bv,s))
                plt.clf()
            """
tacs=np.array(tacs).T
print tacs.shape, t.shape
plt.subplot(312)
#plt.plot(t,np.std(tacs,axis=1))
#plt.plot(t,np.min(tacs,axis=1))
plt.boxplot(tacs.T)
plt.subplot(313)
plt.boxplot(np.array(bad_tacs))

#plt.plot(t,maximums,'o-')

plt.show()



