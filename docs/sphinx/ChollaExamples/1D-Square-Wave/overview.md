# 1D Square Wave
This test initializes a square wave density pertubation. The setup consists of an initial density and pressure of 1.0 and 0.01, respectively. A square wave is initialized with amplitude 1.5. Gamma is set to 1.666666666666667. This test was performed with the hydro build ({repository-file}`config/make.type.hydro`) and Van Leer integrator. Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Square_Wave()`. 

The parameter file can be found at: {repository-file}`examples/1D/square_wave.txt`

## Parameter file:
```
#
# Parameter File for square wave test
#

################################################
# number of grid cells in the x dimension
nx=100
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=1.0
# time interval for output
outstep=0.01
n_hydro=1
# name of initial conditions
init=Square_Wave
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
# Parameters for square wave 
# initial density 
rho=1.0
# velocity in the x direction 
vx=1.0
# velocity in the y direction
vy=0
# velocity in the z direction
vz=0
# initial pressure 
P=0.01
# relative amplitude of overdense region 
A=1.5
# value of gamma
gamma=1.666666666666667
```
Upon completion, you should obtain 101 output files. The evolution of the density is shown below. Pressure is constant to the $10^{-14}$ level. Examples of how to extract and plot data can be found in the [General 1D Plotting Example](../../PythonExamples/1D-plotting.md).    


:::{video} square-docs.mp4
    :width: 640
    :height: 480
    :autoplay:
    :loop:
    :align: center
:::

We see a square waveform of amplitude 1.5 propagating rightwards.  

If the wave is left to propagate for an extended period of time, we observe a rapid breakdown in the structure. This breakdown is much faster with the Van Leer integrator than with the Simple integrator:

Van Leer:

<img src="https://github.com/user-attachments/assets/52248416-8606-43e8-bac5-8d4179581785" width="682" height="452" />

