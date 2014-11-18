__author__ = 'denest'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from perfusion import express



ranges={'bv':(0.1,0.9,0.05),
        'a':(0.1,70,0.5),
        'loc':(0,40,0.5),
        'scale':(0.1,70,0.5),
        'bf_mask':(20,300),
        'mtt_mask':(1,60)
}
fname='pars_a-{a[0]}-{a[1]}-{a[2]}_loc-{loc[0]}-{loc[1]}-{loc[2]}_scale-{scale[0]}-{scale[1]}-{scale[2]}_bv-{bv[0]}-{bv[1]}-{bv[2]}'\
       .format(**ranges)
"""

pars_subset = express.combine_pars(bv=np.arange(*ranges['bv']),
                                   a=np.arange(*ranges['a']),
                                   loc=np.arange(*ranges['loc']),
                                   scale=np.arange(*ranges['scale']))
print pars_subset['a'].shape
pars_subset = dict([[k,list(v)] for k,v in pars_subset.items()])
print 'converted'
with open(fname, 'wb') as fp:
    json.dump(pars_subset, fp)
"""    
print 'saved'
with open(fname, 'rb') as fp:
    pars_subset = json.load(fp)
    print 'converted'
    pars_subset = dict([[k,np.array(v)] for k,v in pars_subset.items()])
print 'loaded'
ranges['bf_mask']=(20,300)
ranges['mtt_mask']=(20,30)

pars_subset=express.filter_pars(pars_subset,mtt_range=ranges['mtt_mask'],bf_range=np.array(ranges['mtt_mask'])/(100.*60))

rc_curves=express.calc_rc_big(pars_subset,0.1)

#
outfile = file('rc_a-{a[0]}-{a[1]}-{a[2]}_loc-{loc[0]}-{loc[1]}-{loc[2]}_scale-{scale[0]}-{scale[1]}-{scale[2]}_bv-{bv[0]}-{bv[1]}-{bv[2]}_\
bf-{bf_mask[0]}-{bf_mask[1]}_mtt-{mtt_mask[0]}-{mtt_mask[1]}'\
               .format(**ranges),
               'w')
np.save(outfile,rc_curves)
outfile.close()
outfile = file('pars_a-{a[0]}-{a[1]}-{a[2]}_loc-{loc[0]}-{loc[1]}-{loc[2]}_scale-{scale[0]}-{scale[1]}-{scale[2]}_bv-{bv[0]}-{bv[1]}-{bv[2]}_\
bf-{bf_mask[0]}-{bf_mask[1]}_mtt-{mtt_mask[0]}-{mtt_mask[1]}'\
               .format(**ranges),
               'w')
np.save(outfile,pars_subset)
outfile.close()



"""
plt.subplot(211)
plt.plot(np.arange(0,100,0.1)[...,None],rc_curves,'k')
plt.subplot(212)
pdfs=express.calc_pdfs(pars_subset,0.1)
plt.plot(np.arange(0,100,0.1)[...,None],pdfs,'r')


plt.show()
#tacs = express.calc_tissue_tacs_mrx_3gamma(input_tac_smoothed, params = pars_subset, time_steps = new_time_steps, time_subset = times)

"""