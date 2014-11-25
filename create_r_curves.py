__author__ = 'denest'
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from perfusion import express


ranges={'bv':(0.1,0.9,0.05),
        'a':(0.01,100,0.2),
        'b':(0.01,100,0.2),
        'loc':(-20,20,1),
        'scale':(5,60,1),
        'bf':(20,300),
        'mtt':(10,15),
        'maxtime':100,
        'dist':'beta'
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
    """
    p=express.combine_pars(bv=np.arange(*ranges['bv']),
                           a=np.arange(*ranges['a']),
                           loc=np.arange(*ranges['loc']),
                           scale=np.arange(*ranges['scale']),
                           )
    print 'pars combined'
    p=express.filter_pars_md(p,
                      mtt_range=ranges['mtt'],
                      bf_range=np.array(ranges['bf'])/(100.*60),
                      time_max=ranges['maxtime'],
                      dist=ranges['dist'])
    """
    a_range=rang
    b_range=np.concatenate([np.arange(0.01,2,0.05),np.arange(2,10,0.2),np.arange(10,60,1)])

    p=express.combine_pars(pars_to_combine=dict(bv=np.arange(*ranges['bv']),
                                               a=a_range,
                                               b=b_range,
                                               loc=np.arange(*ranges['loc']),
                                               scale=np.arange(*ranges['scale'])),
                          mtt_range=ranges['mtt'],
                          bf_range=np.array(ranges['bf'])/(100.*60),
                          time_max=ranges['maxtime'],
                          dist=ranges['dist'])
    print 'pars filtered'
    p_fname=express.gen_fname(prefix='pars',**ranges)

    express.save_pars(p,p_fname)


#Load and save all subsets of pars
def create_pars_subset():
    p_all = express.load_pars(express.gen_fname(prefix='pars',**ranges))
    print express.gen_fname(prefix='pars',**ranges)
    s=0
    for mtt_fr,mtt_to in zip(range(1,60,5),range(6,62,5)):
        ranges['mtt']=(mtt_fr,mtt_to)
        pars_subset=express.filter_pars_md(p_all,
                                        mtt_range=ranges['mtt'],
                                        dist=ranges['dist'])
        s+=pars_subset['a'].shape[0]
        pars_fname = express.gen_fname(prefix='pars',**ranges)
        express.save_pars(pars_subset,pars_fname)
    print s
##Create and save all residue curves
def create_rc_subset(pars_folder, output_folder):
    for p,d,f in os.walk(pars_folder):
        for i in f:
            pars_subset=express.load_pars(os.path.join(pars_folder,i))
            ranges = express.fname_to_range(i)
            ranges['dist']='gamma' #temporary
            rc_fname = express.gen_fname(prefix='rc',full=True,**ranges)

            print ranges
            #express.filter_pars_md(pars_subset,time_max=100,dist='gamma')
            rc = express.calc_rc_big(pars_subset,time_res=0.1)
            express.save_rc(rc,os.path.join(output_folder,rc_fname))

    """
    print express.fname_to_range(i)
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
    """

def read_rc(pathto):
    for p,d,f in os.walk(pathto):
        for i in f:
            print express.fname_to_range(i)


if __name__=='__main__':
    FILES_FOLDER = '/home/denest/PERFetc_files'
    PARS_FOLDER = 'pars'
    RC_FOLDER = 'rc'
    root_folder = os.path.abspath(FILES_FOLDER)
    pars_folder = os.path.join(root_folder,PARS_FOLDER)
    rc_folder = os.path.join(root_folder,RC_FOLDER)
    #create_all()
    #create_pars_subset()
    create_rc_subset(pars_folder,rc_folder)
    #read_rc(os.path.abspath('rc_100_1'))