#!/usr/bin/env python3
# Example python plotting script for the 1D Shu Osher Test


import h5py
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.default']='regular'
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
import matplotlib.pyplot as plt

dnamein='./output-p/' #path to file with simulation data
dnameout='./pngs-p/' #created file for pngs

DE = 0 # dual energy flag - 1 if the test was run with dual energy
i = 1  # output file number

for i in range(0,2):
    f = h5py.File(dnamein+str(i)+'/'+str(i)+'.h5.0', 'r')
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
    time = head['t'][0]

    p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
    ge  = p/d/(gamma - 1.0)

    #plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = (6,8),sharex=True)

    #density
    plt.axis([0, nx, 0, 6])
    ax1.plot(d, 'o', markersize=2, color='black')
    ax1.set_ylabel('Density')
    #velocity
    plt.axis([0, nx, -1, 4]) 
    ax2.plot(vx, 'o', markersize=2, color='black')
    ax2.set_ylabel('Velocity')
    #position
    plt.axis([0, nx, 0, 15])
    ax3.plot(p, 'o', markersize=2, color='black')
    ax3.set_ylabel('Pressure')
    ax3.set_xlabel('Position')

    #add timestamp
    plt.suptitle(f't= {time:.2f}',fontsize=12)
    
    #save png to file
    plt.savefig(dnameout+str(i)+".png", dpi=300);
    plt.close(fig)
    