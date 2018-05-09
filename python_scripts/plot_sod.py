# Example python plotting script for the 1D Sod Shock Tube test

import h5py
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.default']='regular'
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
import matplotlib.pyplot as plt

dnamein='./hdf5/'
dnameout='./png/'

DE = 0 # dual energy flag - 1 if the test was run with dual energy
i = 1 # output file number

f = h5py.File('./hdf5/'+str(i)+'.h5', 'r')
head = f.attrs
nx = head['dims'][0]
gamma = head['gamma'][0]
d  = np.array(f['density']) # mass density
mx = np.array(f['momentum_x']) # x-momentum
my = np.array(f['momentum_y']) # y-momentum
mz = np.array(f['momentum_z']) # z-momentum
E  = np.array(f['Energy']) # total energy density
vx = mx/d
vy = my/d
vz = mz/d
if DE:
  e  = np.array(f['GasEnergy'])
  p  = e*(gamma-1.0)
  ge = e/d
else: 
  p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
  ge  = p/d/(gamma - 1.0)

fig = plt.figure(figsize=(6,6))
ax1 = plt.axes([0.1, 0.6, 0.35, 0.35])
plt.axis([0, nx, 0, 1.1])
ax1.plot(d, 'o', markersize=2, color='black')
plt.ylabel('Density')
ax2 = plt.axes([0.6, 0.6, 0.35, 0.35])
plt.axis([0, nx, -0.1, 1.1])
ax2.plot(vx, 'o', markersize=2, color='black')
plt.ylabel('Velocity')
ax3 = plt.axes([0.1, 0.1, 0.35, 0.35])
plt.axis([0, nx, 0, 1.1])
ax3.plot(p, 'o', markersize=2, color='black')
plt.ylabel('Pressure')
ax4 = plt.axes([0.6, 0.1, 0.35, 0.35])
plt.axis([0, nx, 1.5, 3.7])
ax4.plot(ge, 'o', markersize=2, color='black')
plt.ylabel('Internal Energy')

plt.savefig(dnameout+str(i)+".png", dpi=300);
plt.close(fig)
