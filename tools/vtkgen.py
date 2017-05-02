import numpy as np
import h5py
import easygui as eg

flname=eg.fileopenbox(default="*.h5")
fl=h5py.File(flname)
k=fl['/fields/k'].value
u=fl['/fields/u'].value
#u=u['real']+1j*u['imag']

flname=eg.filesavebox(default="*.vtk")
fid = open (flname, "w");
fid.write("# vtk DataFile Version 3.0\n");
fid.write("patates\n");
fid.write("ASCII\nDATASET UNSTRUCTURED_GRID\n");
fid.write("POINTS %i FLOAT\n" % (k.shape[1]*2));
a=eg.codebox(title=["enter the range"],text="#nt=9999\nnt=np.arange(8000,10000)")
exec(a)
N=k.shape[1]
for l in range(N):
    kx=k[0,l];
    ky=k[1,l];
    kz=k[2,l];
    kk=np.sqrt(kx**2+ky**2+kz**2);
    th=np.arccos(kz/kk);
    if(kx==0):
        ph=0;
    else:
        ph=np.arctan2(ky,kx);
    kkx=np.log10(kk)*np.sin(th)*np.cos(ph);
    kky=np.log10(kk)*np.sin(th)*np.sin(ph);
    kkz=np.log10(kk)*np.cos(th);
    fid.write("% 4.4f % 4.4f % 4.4f\n" %(kkx,kky,kkz));
    fid.write("% 4.4f % 4.4f % 4.4f\n" %(-kkx,-kky,-kkz));
    kn=np.sqrt(kkz**2+kky**2+kkz**2)

fid.write("CELLS %i %i\n"%(N*2,4*N));

for l in range(2*N):
    fid.write("1 %i\n"%l)

fid.write("CELL_TYPES %i\n"%(2*N));
for l in range(2*N):
    fid.write("1\n");

fid.write("POINT_DATA %i\n"%(2*N))
fid.write("SCALARS sample_scalars float 1\n");
fid.write("LOOKUP_TABLE default\n");

for l in range(N):
    kn=np.sqrt(np.sum(k[:,l]**2))
    fid.write("%f\n"%np.log10(np.mean(np.sum(np.abs(u[nt,:,l])**2,1))/kn));
    fid.write("%f\n"%np.log10(np.mean(np.sum(np.abs(u[nt,:,l])**2,1))/kn));
fid.close()
