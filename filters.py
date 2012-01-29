__author__ = 'denis'
import numpy as np

def std(image,num):
    sh=np.shape(image)[0]
    for parts in np.arange(num)[::-1]:
        print sh%parts,parts
        if sh%parts == 0:
            partsnum=parts
            break
    it=np.linspace(0,sh,partsnum+1)
    print it
    tips=np.std(image[it[0]:it[1]],axis=-1)
    for ind,val in enumerate(it[2:]):
        tipsplus=np.std(image[it[ind+1]:val],axis=-1)
        tips=np.append(tips,tipsplus,0)
        print it[ind+1],val
    return tips
