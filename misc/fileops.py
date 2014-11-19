__author__ = 'denest'
import os
import numpy as np

def read_rc_gen_settings():

    out={}
    with file('rc_gen_settings','rb') as f:
        for s in f.readlines():
            s=s.strip('\n')
            if s.startswith('-'):
                parsname=s.lstrip('-')
                out[parsname]={}
            else:
                distpar_name,distpar_val = s.split(':')
                try:
                    distpar_val = tuple([map(float,i.split(',')) for i in distpar_val.split(';')])
                    if len(distpar_val)==1:
                        distpar_val=distpar_val[0]
                except ValueError:
                    pass
                out[parsname][distpar_name]=distpar_val
    return out




def gen_ranges(**args):
    return np.concatenate([np.arange(i) for i in args])
print read_rc_gen_settings()