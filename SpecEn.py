#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:05:00 2017

@author: ogurcan
"""

import numpy as np
import h5py as h5
import matplotlib.pylab as plt
sh0=1
flname="nsout.h5"
fl=h5.File(flname,"r")
u=fl['fields/u'].value
k=fl['fields/k'].value
t=fl['fields/t'].value
fl.close()
#the base node number for the nth shell
def nbase(n):
    return (n%2)*2*int(2*(1/2-sh0))+n*8

#size of the polyhedron, given its base node number
def sz4base(n):
    return int(8-4*(1/2-(n+(1-sh0)*10)%16/nbase(1)))

Ns=int(u.shape[2]/8)

ntr=np.array((np.arange(7100,7200),np.arange(8300,8400),np.arange(9300,9400)))

En=np.zeros((ntr.shape[0],Ns))
kn=np.zeros(Ns)
for l in range(Ns):
    nb=nbase(l)
    sz=sz4base(nb)
    kn[l]=np.sqrt(np.sum(k[:,nb]**2,0))
    lr=np.arange(nb,nb+sz)
    for j in range(ntr.shape[0]):
        En[j,l]=np.mean(np.sum(np.abs(u[:,:,lr])**2,(1,2))[ntr[j,:]])/sz/kn[l]
plt.loglog(kn,En.T,'x-',kn,1e-1*kn**(-5/3))
