#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:05:00 2017

@author: ogurcan
"""

import numpy as np
import h5py as h5
import matplotlib.pylab as plt
import easygui as eg

msgtxt=""
flnames=[]
codes=[]
cont=True
n=0
#the base node number for the nth shell
def nbase(n,sh0):
    return (n%2)*2*int(2*(1/2-sh0))+n*8

#size of the polyhedron, given its base node number
def sz4base(n,sh0):
    return int(8-4*(1/2-(n+(1-sh0)*10)%16/nbase(1,sh0)))

def compute_spec(flname,code):
    fl=h5.File(flname,"r")
    u=fl['fields/u'].value
    k=fl['fields/k'].value
#    t=fl['fields/t'].value
    fl.close()
    exec(code,globals())
    ktest=np.sum(k[:,0:10]**2,0)
    sh0=1
    if(ktest[9]==ktest[0]):
        sh0=0
    Ns=int(u.shape[2]/8)
    En=np.zeros(Ns)
    kn=np.zeros(Ns)
    for l in range(Ns):
        nb=nbase(l,sh0)
        sz=sz4base(nb,sh0)
        kn[l]=np.sqrt(np.sum(k[:,nb]**2,0))
        lr=np.arange(nb,nb+sz)
        En[l]=np.mean(np.sum(np.abs(u[:,:,lr])**2,(1,2))[nt])/sz/kn[l]
    return En,kn

while(cont):
    flname=eg.fileopenbox(default="*.h5")
    code=eg.codebox(title=["enter the range in python format"],text="#nt=9999\nnt=np.arange(8000,10000)")
    msgtxt+="spectrum "+str(n)+"\n----------------\n"+flname+"\n"+code+"\n----------------\n"
    flnames.append(flname)
    codes.append(code)
    yn=eg.ynbox(msg=msgtxt,choices=('[<F1>]Add Another One', '[<F2>]Continue'))
    n+=1
    if not yn:
        cont=0

#ccode=eg.codebox(title=["enter the list of colors"],"colors=['b','k','r','g']")
#prop_cycle = plt.rcParams['axes.prop_cycle']
#colors = prop_cycle.by_key()['color']
colors=['r','b','g','rosybrown','silver']
Enm=[]
knm=[]
for n in range(len(flnames)):
    En,kn=compute_spec(flnames[n],codes[n])
    Enm.append(En)
    knm.append(kn)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig,ax=plt.subplots()
nmax=0
shapemax=0
for n in range(len(flnames)):
    kn=knm[n]
    if(kn.shape[0]>shapemax):
        shapemax=kn.shape[0]
        nmax=n
    En=Enm[n]
    ax.loglog(kn,En.T,'s-',markersize=3,color=colors[n])
kn=knm[nmax]
ax.loglog(kn,1e-1*kn**(-5/3),'k')
mk=kn[int(len(kn)/2)]
plt.text(mk*1.1,1e-1*mk**(-5/3)*1.1,'$k^{-5/3}$',fontsize=16)
plt.xticks(10.0**np.arange(0,int(np.log10(kn[-1]))+1),fontsize=14)
plt.yticks(10.0**np.arange(int(np.log10(np.min(Enm[nmax])))-1,int(np.log10(np.max(Enm[nmax])))+1,2),fontsize=14)
ax.minorticks_off()
plt.title(r"$E(k)$",fontsize=16)
plt.xlabel(r"$k$",fontsize=16)