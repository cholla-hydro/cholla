# KHD Plotting Script 

The output of this script should be 199 pngs.

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

dnamein='./directory_name/' # directory where the file is located
dnameout='./png_output_directory/' # directory where the plot will be saved

DE = 0
i = 0

mp = 1.672622e-24 # mass of hydrogren atom, in grams
kb = 1.380658e-16 # boltzmann constant in ergs/K

for i in range(0,200):
    f = h5py.File(dnamein+str(i)+'/'+str(i)+'.h5.0', 'r')
    head = f.attrs


    gamma = head['gamma'] # ratio of specific heats
    t  = head['t'] # time of this snapshot, in kyr
    nx = head['dims'][0] # number of cells in the x direction
    ny = head['dims'][1] # number of cells in the y direction
    nz = head['dims'][2] # number of cells in the z direction
    dx = head['dx'][0] # width of cell in x direction
    dy = head['dx'][1] # width of cell in y direction
    dz = head['dx'][2] # width of cell in z direction
    l_c = head['length_unit']
    t_c = head['time_unit']
    m_c = head['mass_unit']
    d_c = head['density_unit']
    v_c = head['velocity_unit']
    e_c = head['energy_unit']
    p_c = e_c # pressure units are the same as energy density units, density*velocity^2/length^3
    
    d  = f['density'][:]
    p = f['pressure'][:]

    time = head['t'][0]

    mu = 1.0 # mean molecular weight (mu) of 1
   
   # d = d*d_c # to convert from code units to cgs, multiply by the code unit for that variable
    
    n = d/(mu*mp) # number density, particles per cm^3


    image = plt.imshow(p.T,origin='lower',cmap='viridis', vmin=1.8, vmax=2.8)

    cb = plt.colorbar(image, ticks=np.arange(1.8,2.8,0.2),label='pressure')

    plt.suptitle(f't={time:.2f}',fontsize=12)

    plt.savefig(dnameout + str(i)+".png", dpi=300)  # You can adjust the dpi for quality
    plt.close()
    