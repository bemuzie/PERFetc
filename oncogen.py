# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:44:53 2013

@author: КТ2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from scipy import signal
from scipy.ndimage import filters

roi=np.zeros([100,100])

class Tumor():
    def __init__(self,matrixsize,tum_size,tumorosity):
        if matrixsize%2 == 0:
            raise ValueError()
        self.matrix = np.zeros ([matrixsize,matrixsize])
        self.conturs= np.zeros ([matrixsize,matrixsize])
        self.ms=matrixsize
        self.distances = self.matrix
        self.grow(np.array([matrixsize/2,matrixsize/2]),tum_size)
        
        for i in range(tumorosity):
            self.new_one(tum_size)

    def grow (self,startpoint,radius):
        x,y=startpoint
    
        log_mrx=np.ogrid[0:self.ms,0:self.ms]-startpoint
        
        euclid_dist=np.sqrt(log_mrx[0]**2+log_mrx[1]**2)
        self.matrix[euclid_dist<radius]=1
    def new_one(self,maxsize=10):
        x,y=[np.random.randint(0,self.ms),np.random.randint(0,self.ms)]
        if self.matrix[x,y]==1:
            self.grow(np.array([x,y]),np.random.randint(1,maxsize))
        else:
            self.new_one()
    def show_tumor(self,data1,data2):
        p1=plt.subplot(121)
        p1.imshow(data1/np.average(data1),cmap = 'gray')
        #p1.colorbar()
        p2=plt.subplot(122)
        p2.imshow(data2/np.average(data2),cmap = 'gray')
        #p2.colorbar()

        plt.show()
    def make_distanses(self):
        self.matrix_1=self.matrix
        
        for i in range(50):
            self.matrix_1=filters.generic_filter(self.matrix_1,self.erode,3)
            self.distances+=self.matrix_1
            print i
        self.conturs[self.distances==1 OR self.distances==2 OR self.distances==3 OR self.distances==12]=1
        
        self.conturs[self.conturs==0]=np.nan
    def make_perfusion(self):
        dist=np.copy(self.distances)

        
        self.perf_map=2/(1+np.exp(dist+np.random.normal(0,20,dist.shape))**0.035)
        
        #self.perf_map/=np.average(self.perf_map)

    def erode(self,data):
        d=data[5]
        if d==1 and 0 in data:
            return 0
        else:
            return d
    
        
    
    

a=Tumor(101,20,50)
tumors = []

cm = plt.get_cmap('bwr')
#cm.set_bad(color = 'k', alpha = 1)

for i in range(2):
    a=Tumor(151,np.random.randint(5,40),40)
    a.make_distanses()
    a.make_perfusion()
    p=plt.subplot(5,5,i+1)
    print np.average(a.perf_map),np.min(a.perf_map),np.max(a.perf_map)

    p.imshow(filters.gaussian_filter(a.perf_map,2),alpha=1,cmap='gist_ncar',norm=mcol.Normalize(vmin=0.5,vmax=1.3))
    p.imshow(a.conturs,alpha=1,cmap=cm)
    
    plt.axis('off')

plt.show()
