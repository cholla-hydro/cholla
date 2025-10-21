# 1D Sound Wave
This test initializes a compression/rarefaction wave across the grid. The setup consists of an initial density and pressure of 1.0 and 0.6, respectively. A sound wave is initialized as a sinusoidal wave with amplitude 1e-4 and wavelength of 1.0. Gamma is set to 1.666666666666667. This test was performed with the hydro build (`cholla/builds/make.type.hydro`) and Van Leer integrator. Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Sound_Wave()`. 

## Parameter file: (modified from`cholla/examples/1D/sound_wave.txt`)
Modified to add yl_bcnd, yu_bcnd, zl_bcnd, and zu_bcnd=0
```
#
# Parameter File for sound wave test
#

################################################
# number of grid cells in the x dimension
nx=128
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=0.05
# time interval for output
outstep=0.01
# name of initial conditions
init=Sound_Wave
# size of domain
xmin=0.0
ymin=0.0
zmin=0.0
xlen=1.0
ylen=1.0
zlen=1.0
# type of boundary conditions
xl_bcnd=1
xu_bcnd=1
yl_bcnd=0
yu_bcnd=0
zl_bcnd=0
zu_bcnd=0
# path to output directory
outdir=./

#################################################
# Parameters for linear wave problems
# initial density
rho=1.0
# velocity in the x direction
vx=0
# velocity in the y direction
vy=0
# velocity in the z direction
vz=0
# initial pressure
P=0.6
# amplitude of perturbing oscillations
A=1e-4
# value of gamma
gamma=1.666666666666667
```
Upon completion, you should obtain five output files. By changing the outstep to 0.005, we can obtain the evolution of the pressure and density (here at 4 fps).  Examples of how to extract and plot data can be found in `cholla/python_scripts/plot_sod.ipynb`.  

:::{video} squarewave-docs.mp4
    :width: 640
    :height: 480
    :autoplay:
    :loop:
    :align: center
:::
