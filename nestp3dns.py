# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:54:02 2016

@author: ogurcan
"""
import numpy as np
import h5py as h5
from scipy.integrate import ode
import time

# wheter the zeroth shell is an icosahedron.
# i.e. sh=1 means icosahedron, sh=0 means dodecahedron.

sh0=1
flname="nsout.h5"     # filename for output (and input if we continue).
wecontinue=False            # do we continue from an existing file.
t0=0.0;                     # t0 will come from the file if we continue.
t1=1000.0;                  # tmax.
dt=0.1;                     # time step for output.
Nmax=40;                    # number of shells.
k0=1.0;                     # beginning of the wavenumber range.
nu=1e-6;                    # kinematic viscosity.

# list of links for an icosahedron
links_ico=np.array([
    [[[10,10],[11,11],[7,12],[8,13],[9,14]],[[5,10],[6,11],[7,12],[8,13],[9,14]],[[15,10],[16,11],[17,7],[18,8],[19,9]]],
    [[[3,10],[4,14],[11,15],[6,7],[8,19]],[[1,10],[3,14],[18,15],[12,7],[16,19]],[[11,3],[13,4],[8,11],[2,6],[6,8]]],
    [[[5,10],[4,11],[9,15],[7,16],[6,8]],[[4,10],[2,11],[17,15],[19,16],[13,8]],[[14,5],[12,4],[7,9],[9,7],[3,6]]],
    [[[1,11],[5,12],[10,16],[8,17],[6,9]],[[0,11],[3,12],[18,16],[15,17],[14,9]],[[10,1],[13,5],[8,10],[5,8],[4,6]]],
    [[[6,5],[1,13],[2,12],[9,18],[11,17]],[[10,5],[4,13],[1,12],[16,18],[19,17]],[[0,6],[14,1],[11,2],[6,9],[9,11]]],
    [[[6,6],[7,18],[2,14],[3,13],[10,19]],[[11,6],[15,18],[0,14],[2,13],[17,19]],[[1,6],[5,7],[10,2],[12,3],[7,10]]]])

# list of links for a dodecahedron
links_dod=np.array([
    [[[15,6],[11,7],[14,8]],[[4,6],[9,7],[11,8]],[[10,15],[3,11],[5,14]]],
    [[[16,6],[12,8],[10,9]],[[5,6],[10,8],[7,9]],[[11,16],[4,12],[1,10]]],
    [[[17,6],[13,9],[11,10]],[[1,6],[11,9],[8,10]],[[7,17],[5,13],[2,11]]],
    [[[18,6],[14,10],[12,11]],[[2,6],[7,10],[9,11]],[[8,18],[1,14],[3,12]]],
    [[[19,6],[13,7],[10,11]],[[3,6],[10,7],[8,11]],[[9,19],[4,13],[2,10]]],
    [[[8,7],[7,8],[10,4]],[[5,7],[3,8],[6,4]],[[11,8],[9,7],[0,10]]],
    [[[9,8],[8,9],[11,5]],[[1,8],[4,9],[6,5]],[[7,9],[10,8],[0,11]]],
    [[[12,1],[5,9],[9,10]],[[6,1],[2,9],[5,10]],[[0,12],[8,5],[11,9]]],
    [[[13,2],[6,10],[5,11]],[[6,2],[3,10],[1,11]],[[0,13],[9,6],[7,5]]],
    [[[6,7],[14,3],[7,11]],[[2,7],[6,3],[4,11]],[[8,6],[0,14],[10,7]]]])

#spherical positions of points of an icosahedron.
def icosp():
    theta=np.hstack([0, np.tile(np.pi/2-np.arctan(1/2),5),np.pi,np.tile(np.pi/2+np.arctan(1/2),5)]);
    phi=np.hstack([0,np.arange(0,2*np.pi,2*np.pi/5),0,np.mod(np.arange(0,2*np.pi,2*np.pi/5)+np.pi,2*np.pi)]);
    return theta,phi;

#spherical positions of points of a dodecahedron.
def dodsp():
    ph=(1+np.sqrt(5))/2;
    alp=np.array([np.arcsin(ph/np.sqrt(3))-np.arccos(ph/np.sqrt(ph+2)),np.arctan(2*ph**2)]);
    eta=np.arange(np.pi/5,2*np.pi,2*np.pi/5);
    theta=np.reshape(np.transpose(np.reshape(np.transpose(np.tile(np.hstack([alp,np.pi-alp]),[5,1])),[20,1])),20);
    phi=np.reshape(np.hstack([np.tile(eta,[1,2]),np.tile(eta-np.pi,[1,2])]),20);
    return theta,phi;

#the base node number for the nth shell
def nbase(n):
    return (n%2)*2*int(2*(0.5-sh0))+n*8

#size of the polyhedron, given its base node number
def sz4base(n):
    return (8-4*(0.5-(n+(1-sh0)*10)%16/nbase(1))).astype(int)

#get the list of (n',l') and (n'',l'') pairs for a given (n,l)
def get_nls(n,l):
    if(n%2==sh0):
        nlp=np.zeros((2,9,2),dtype=int)
        nlp[0,0:3,0]=n-2
        nlp[0,0:3,1]=n-1
        nlp[0,3:6,0]=n-1
        nlp[0,3:6,1]=n+1
        nlp[0,6:9,0]=n+1
        nlp[0,6:9,1]=n+2
        nlp[1,:,:]=links_dod[l,:,:,:].reshape(nlp[1,:,:].shape)
    else:
        nlp=np.zeros((2,15,2),dtype=int)
        nlp[0,0:5,0]=n-2
        nlp[0,0:5,1]=n-1
        nlp[0,5:10,0]=n-1
        nlp[0,5:10,1]=n+1
        nlp[0,10:15,0]=n+1
        nlp[0,10:15,1]=n+2
        nlp[1,:,:]=links_ico[l,:,:,:].reshape(nlp[1,:,:].shape)
    return nlp

# get the list of node pairs that interact with the node (n,l) and 
# wheter or not  they are conjugated (i.e. cp and cpp)
def get_interacting_nodes(n,l):
    npl=get_nls(n,l)
    nnp=nbase(npl[0,:,0])
    lp=npl[1,:,0]
    nnpp=nbase(npl[0,:,1])
    lpp=npl[1,:,1]
    szp=sz4base(nnp)
    szpp=sz4base(nnpp)
    cp=((lp-szp)>=0)
    cpp=((lpp-szpp)>=0)
    lp=lp-cp.astype(int)*szp
    lpp=lpp-cpp.astype(int)*szpp
    nlp=nnp+lp
    nlpp=nnpp+lpp
    cp=~cp
    cpp=~cpp
    return nlp,nlpp,cp,cpp

#number of nodes
NLmax=np.floor((Nmax/2)).astype(int)*2*8+(Nmax%2)*(10-((Nmax+1+sh0)%2)*4)
g=np.sqrt((1+np.sqrt(5))/2)
lam=np.sqrt(np.sqrt(5)/3)
th,ph=dodsp()
khatd=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
th,ph=icosp()
khati=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
k=np.zeros((3,NLmax));
links=[]

#constructing the network: the nodes and the links
for n in np.arange(0,Nmax):
    if(n%2==sh0):
        k[:,nbase(n):nbase(n+1)]=khatd[:,0:10]*g**n
    else:
        k[:,nbase(n):nbase(n+1)]=khati[:,0:6]*g**n*lam
    for l in np.arange(0,10-(n+sh0)%2*4):
        nl=nbase(n)+l
        nlp,nlpp,cp,cpp=get_interacting_nodes(n,l)
        inds=np.nonzero((nlp>=0) & (nlpp>=0) & (nlp<NLmax) & (nlpp<NLmax))
        links.append([nlp[inds],nlpp[inds],cp[inds],cpp[inds]]);

#initializing u
kn=np.sqrt(np.sum(k**2,0))
if(wecontinue==True):
    #setting up the output hdf5 file
    fl=h5.File(flname,"r")
    u=fl["fields/u"].value
    tt=fl["fields/t"].value
    fl.close()
    t0=tt[-1]
    unls=u[-1,:,:]
else:
    unls=np.zeros((3,NLmax),dtype=complex)
    unls[:,:]=0.0001+0.0001j;
    #making sure u is divergence free
    unls=unls-(np.einsum("ij,ij->j",k,unls)*k)/kn**2
#setting up the interaction coefficient
Mkpq=np.einsum("kl,ij->kijl",k,np.eye(3))+np.einsum("jl,ik->kijl",k,np.eye(3))-2*np.einsum("il,jl,kl,l->kijl",k,k,k,1/kn**2)
uu0=unls.reshape(unls.shape[0]*unls.shape[1]);
dudt=np.zeros(np.shape(unls),dtype=unls.dtype);

#defining the form of hyperviscosity
Dn=nu*kn**2

#setting up the forcing
Fn=np.zeros((3,NLmax),dtype=unls.dtype);
Fn[:,nbase(3):nbase(4)]=0.01+0.01j
Fn[:,nbase(4):nbase(5)]=0.01-0.01j
Fn=Fn-(np.einsum("ij,ij->j",k,Fn)*k)/kn**2

#define cj as "conjugate of y if x is True"
cj=lambda x,y: np.conj(y) if x else y
cj=np.vectorize(cj)

#the equation
def func(t,y,par):
    u=y.reshape((3,NLmax));
    for n in np.arange(0,NLmax):
        nlp=links[n][0]
        nlpp=links[n][1]
        cp=links[n][2]
        cpp=links[n][3]
        up=u[:,nlp]
        upp=u[:,nlpp]
        dudt[:,n]=Fn[:,n]-Dn[n]*u[:,n]-1j*np.einsum("kij,kl,jl->i",Mkpq[:,:,:,n],cj(cp,up),cj(cpp,upp))
        dydt=np.reshape(dudt,NLmax*3);
    return dydt;

#setting up the output hdf5 file
fl=h5.File(flname,"w")
grp=fl.create_group("fields")
grp.create_dataset("k",data=k)
if(wecontinue==True):
    i=u.shape[0]
    ures=grp.create_dataset("u",(i,3,NLmax),maxshape=(None,3,NLmax),dtype=complex)
    tres=grp.create_dataset("t",(i,),maxshape=(None,),dtype=float)
    ures[:,:,:]=u;
    tres[:]=tt;
else:
    i=0;
    ures=grp.create_dataset("u",(1,3,NLmax),maxshape=(None,3,NLmax),dtype=complex)
    tres=grp.create_dataset("t",(1,),maxshape=(None,),dtype=float)

#initializing the ode solver
r = ode(func, 0).set_integrator('zvode', method='bdf', with_jacobian=False,atol=1e-14,rtol=1e-8,nsteps=1E6);
r.set_initial_value(uu0, t0).set_f_params(0.0).set_jac_params(0.0);
#the main loop where the solution is obtained and written to the hdf5 file:
ct=time.time()
while r.successful() and r.t < t1:
    print("t=",r.t);
    r.integrate(r.t+dt)
    ures.resize((i+1,3,NLmax))
    tres.resize((i+1,))
    ures[i,:,:]=np.reshape(r.y,(3,NLmax))
    tres[i]=r.t
    fl.flush()
    i=i+1;
    print(time.time()-ct,"seconds elapsed.")
fl.close()
