#-*- codding utf-8 -*-
__author__ = 'ct'

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


class bottle():
    def __init__(self,volume,name,distribution=stats.uniform,distargs=[10,5]):
        self.name=name
        self.inflow_bottles={}
        self.outflow_bottles={}
        self.volume=volume
        self.tempvolume=0
        self.distribution=distribution
        self.distargs=distargs

    def addout(self,bottle,flow):
        #adding compartment for outflow.sum of all flows should be 1

        self.outflow_bottles[bottle]=flow
        if not self in bottle.inflow_bottles:
            bottle.addin(self,flow)
    def addin(self,bottle,flow):
        self.inflow_bottles[bottle]=flow
        if not self in bottle.outflow_bottles:
            bottle.addout(self,flow)
    def flow(self,times):

        self.tempvolume=np.sum([self.volume*self.outflow_bottles[btl] for btl in self.outflow_bottles])
        print self.name, "tempvolume=",self.tempvolume
        self.volume-=self.tempvolume

        for btl in self.outflow_bottles:
            if times==0:
                break
            self.volume+=btl.tempvolume
            print "At time", times," ", self.name, "has ", self.volume
            btl.flow(times-1)
    def setoutconc(self,time,timeres,nullout):
        self.t=np.linspace(float(time)/timeres,time,timeres)
        self.outconc=nullout
        self.outf=self.distribution.pdf(self.t,*self.distargs)
        self.outf/=np.sum(self.outf)

    def outflow(self):
        if self.inflow_bottles=={}:
            pass
        else:
            self.outconcnew=np.convolve(self.outf, np.sum([btl.outconc
                                                          for btl in self.inflow_bottles], axis=0)) [:self.t.size]
            diff=(self.outconcnew-self.outconc)
            self.outconc=self.outconcnew
            if np.sum(diff*diff)>0.001:
                for btl in self.inflow_bottles:
                    btl.outflow()
