import h5py as h5
import numpy as np
import sys, os

TIME_UNIT = 3.15569e10
LENGTH_UNIT = 3.08567758e21
MASS_UNIT = 1.98847e33
DENSITY_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))
VELOCITY_UNIT = (LENGTH_UNIT/TIME_UNIT)
NB_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))/1.674e-24
ENERGY_UNIT = (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)


n = 0
while(os.path.exists("OUT/%d.h5.0"%n)):

    f = "OUT/%d.h5.0"%n
    d = h5.File( f, 'r')

    t = d.attrs['t'][0]
    
    rho = d['density'][...] 
    vx = d['momentum_x'][...]/rho
    vy = d['momentum_x'][...]/rho
    vz = d['momentum_x'][...]/rho

    nb = rho*NB_UNIT
    xHI = d['HI_density'][...]/(rho)
    xHII = d['HII_density'][...]/(rho)

    U = ENERGY_UNIT*(d['Energy'][...]-0.5*rho*(vx**2+vy**2+vz**2))
    T = U/(1.5*nb*1.38e-16*(xHI+2*xHII))

    print(t*1e3, xHI.mean(), T.mean())

    n += 1
##
