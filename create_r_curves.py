__author__ = 'denest'
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from perfusion import express


ranges={'bv':(0.1,0.9,0.02),
        'a':(0.01,50,0.5),
        'loc':(0,30,0.5),
        'scale':(0.1,70,0.5),
        'bf_mask':(20,300),
        'mtt_mask':(1,60),
        'time_max':100,
        'dist':'gamma'
}
"""
print express.gen_fname(prefix='pars',full=False,**ranges)

pars_subset=express.load_pars(express.gen_fname(prefix='pars',full=False,**ranges))

pars_subset=express.filter_pars(pars_subset,
								mtt_range=ranges['mtt_mask'],
								bf_range=np.array(ranges['bf_mask'])/(100.*60))

rc_curves=express.calc_rc_big(pars_subset,0.1)

pars_fname = express.gen_fname(prefix='pars',**ranges)
rc_fname = express.gen_fname(prefix='rc',**ranges)


express.save_pars(pars_subset,pars_fname)
express.save_pars(rc_curves,rc_fname)
"""
#Create and save all pars
def create_all():
    print ranges
    p=express.combine_pars(bv=np.arange(*ranges['bv']),
                           a=np.arange(*ranges['a']),
                           loc=np.arange(*ranges['loc']),
                           scale=np.arange(*ranges['scale']),
                           )
    print 'pars combined'
    p=express.filter_pars_md(p,
                      mtt_range=ranges['mtt_mask'],
                      bf_range=np.array(ranges['bf_mask'])/(100.*60),
                      time_max=ranges['time_max'],
                      dist=ranges['dist'])
    print 'pars filtered'
    p_fname=express.gen_fname(prefix='pars',**ranges)

    express.save_pars(p,p_fname)


#Load and save all subsets of pars
def create_pars_subset():
    p_all = express.load_pars(express.gen_fname(prefix='pars',full=True,**ranges))
    s=0
    for mtt_fr,mtt_to in zip(range(1,60,5),range(6,62,5)):
        ranges['mtt_mask']=(mtt_fr,mtt_to)
        pars_subset=express.filter_pars(p_all,
                                        mtt_range=ranges['mtt_mask'])
        s+=pars_subset['a'].shape[0]
        pars_fname = express.gen_fname(prefix='pars',full=True,**ranges)
        express.save_pars(pars_subset,pars_fname)
    print s
##Create and save all residue curves
def create_rc_subset():
    s=0
    for mtt_fr,mtt_to in zip(range(1,60,5),range(6,62,5)):
        ranges['mtt_mask']=(mtt_fr,mtt_to)
        pars_fname = express.gen_fname(prefix='pars',full=True,**ranges)
        rc_fname = express.gen_fname(prefix='rc',full=True,**ranges)

        pars_subset=express.load_pars(pars_fname)
        rc=express.calc_rc_big(pars_subset,1)
        express.save_rc(rc,rc_fname)
        print mtt_fr,mtt_to


    print s
def read_rc(pathto):
    for p,d,f in os.walk(pathto):
        for i in f:
            print express.fname_to_range(i)


if __name__=='__main__':
    create_all()
    #create_pars_subset()
    #create_rc_subset()
    #read_rc(os.path.abspath('rc_100_1'))