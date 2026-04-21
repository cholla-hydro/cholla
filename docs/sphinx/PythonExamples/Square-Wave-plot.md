# Square Wave Plotting Script

The output of this script should be 99 pngs.

```python
import h5py
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.default']='regular'
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['xtick.top']=True
matplotlib.rcParams['ytick.right']=True
import matplotlib.pyplot as plt

dnamein='./output/'
dnameout='./pngs/'

DE = 0 # dual energy flag - 1 if the test was run with dual energy
i = 1  # output file number

for i in range(0,100):    
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

    fig = plt.figure(figsize=(6,4))
    plt.axis([0, nx, 0.8,1.8])
    plt.plot(d, 'o', markersize=3, color='blueviolet')
    plt.ylabel('Density')
    plt.xlabel('Position')

    plt.tight_layout(rect=[0,0,1,0.95])

    plt.suptitle(f't = {time:.2f}', fontsize=12)

    plt.savefig(dnameout+str(i)+".png", dpi=300);
    plt.close(fig)


