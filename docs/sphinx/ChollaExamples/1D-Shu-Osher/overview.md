# 1D Shu and Osher test
This test (Shu & Osher, 1989) highlights the ability of a code to resolve small scale smooth flow and shocks simultaneously. Further, it shows how lower resolution solutions can cut off some of the amplitude of maxima due to the slope limiters. Parameters are from Stone et al., 2008, Section 8.1. The test consists of left and right states separated at x = -0.8. On the left, density is set to 3.857143, pressure to 10.33333, and velocity to 2.629369. On the right, density is sinusoidally varying: $\rho(x)$ =  1.0 + 0.2 $\sin(5.0\pi x)$. Pressure is set to 1.0 and the velocity is 0.0. Gamma is set to 1.4. This test is performed with the hydro build (`cholla/builds/make.type.hydro`). Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Shu_Osher()`.  


## Parameter file: (**modified** from `cholla/examples/1D/Shu_Osher.txt`)
Modified to add yl_bcnd, yu_bcnd, zl_bcnd, and zu_bcnd=0. Xmin changed to -1.0 and xlen to 2.0.
```
# Parameter File for the Shu-Osher shock tube test, originally from
#     Shu & Osher, 1989. These parameters are from Stone et al., 2008, Section 8.1
#

######################################
# number of grid cells in the x dimension
nx=200
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=0.47
# time interval for output
outstep=0.47
# value of gamma
gamma=1.4
# name of initial conditions
init=Shu_Osher
# domain properties
xmin=-1.0
ymin=0.0
zmin=0.0
xlen=2.0
ylen=1.0
zlen=1.0
# type of boundary conditions
xl_bcnd=3
xu_bcnd=3
yl_bcnd=0
yu_bcnd=0
zl_bcnd=0
zu_bcnd=0
# path to output directory
outdir=./
```
Upon completion, you should obtain 2 output files. The initial and final density, pressure, and velocity (in code units) of the solution is shown below (pink dots) plotted over high resolution solution with 4000 cells (purple line). Examples of how to extract and plot data can be found in `cholla/python_scripts/plot_sod.ipynb`.  

:::{figure} shu-osher.png

With the diode disabled, this solution does match that of Schneider and Robertson 2015 and Stone et al. 2008, shown below:

:::{figure} reference.png


