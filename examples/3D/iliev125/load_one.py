import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob

TIME_UNIT = 3.15569e10
LENGTH_UNIT = 3.08567758e21
MASS_UNIT = 1.98847e33
#TIME_UNIT = (1e3*3.15569e10)
#LENGTH_UNIT = (13.2*3.08567758e21)
#MASS_UNIT = 1.1289245801680841e+41
DENSITY_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))
VELOCITY_UNIT = (LENGTH_UNIT/TIME_UNIT)
NB_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))/1.674e-24
ENERGY_UNIT = (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)


d = h5.File( sys.argv[1], 'r')
print("t=",d.attrs["t"])
if(len(sys.argv) > 2):
    if(sys.argv[2] == "-l"):
        rho = d['density'][...]
        print(rho.shape)
        for k in d.keys():
            v = d[k][...]
            print(k,v.min(),v.max())
        ##
    else:
        print("Invalid option(s): ",sys.argv[2:])
        sys.exit(1);
    ##
##

print(d['rf1'][32,0,0])

rho = d['density'][...]
vx = d['momentum_x'][...]/rho
vy = d['momentum_x'][...]/rho
vz = d['momentum_x'][...]/rho

nb = rho*NB_UNIT
print(nb.min(),nb.max())

xHI = d['HI_density'][...]/(rho)
xHII = d['HII_density'][...]/(rho)

U = ENERGY_UNIT*(d['Energy'][...]-0.5*rho*(vx**2+vy**2+vz**2))
T = U/(1.5*nb*1.38e-16)

print("xHI=%10.3e xHII=%10.3e T=%10.3e nb=%10.3e"%(xHI.mean(),xHII.mean(),T.mean(),nb.mean()))

fig = plt.figure(figsize=(9.20,3.10))
fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.05)

ax1 = fig.add_subplot(1,4,1)
ax2 = fig.add_subplot(1,4,2)
ax3 = fig.add_subplot(1,4,3)
ax4 = fig.add_subplot(1,4,4)
import matplotlib.colors as col

ax1.imshow(xHI[xHI.shape[0]//2,:,:],cmap="rainbow",vmin=1.0e-3,vmax=1)
ax2.imshow(xHII[xHI.shape[0]//2,:,:],cmap="rainbow",vmin=1.0e-3,vmax=1)
ax3.imshow(d['rf1'][xHI.shape[0]//2,:,:],cmap="rainbow",norm=col.SymLogNorm(1.0e-8,vmin=1.0e-8,clip=1))
ax4.imshow(1e3*nb[xHI.shape[0]//2,:,:],cmap="rainbow",vmin=0.5,vmax=1.5)

plt.show()


