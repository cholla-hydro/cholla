# 1D Sod Shock Tube
This test highlights the ability of a code to resolve shocks and contact discontinuities over a narrow region. Parameters from Sod (1978). The setup consists of a density and pressure of 1.0 for x \< 0 and 0.1 for x \> 0.5. Gamma is set to 1.4. This test was performed with the hydro build (`cholla/builds/make.type.hydro`) and Van Leer integrator. Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Riemann()`. 


The parameter file can be found at {repository-file}`examples/1D/sod.txt`

## Parameter File:
```
#
# Parameter File for 1D Sod Shock tube
#

################################################
# number of grid cells in the x dimension
nx=100
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=0.2
# time interval for output
outstep=0.2
# name of initial conditions
init=Riemann
# domain properties
xmin=0.0
ymin=0.0
zmin=0.0
xlen=1.0
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

#################################################
# Parameters for 1D Riemann problems
# density of left state
rho_l=1.0
# velocity of left state
vx_l=0.0
vy_l=0.0
vz_l=0.0
# pressure of left state
P_l=1.0
# density of right state
rho_r=0.1
# velocity of right state
vx_r=0.0
vy_r=0.0
vz_r=0.0
# pressure of right state
P_r=0.1
# location of initial discontinuity
diaph=0.5
# value of gamma
gamma=1.4
```
Upon completion, you should obtain two output files.The initial and final density, pressure, and velocity (in code units) of the solution is shown below.  Examples of how to extract and plot data can be found in the [General 1D Plotting Example](../../PythonExamples/1D-plotting.md).  

:::{figure} sod-initial-final.png
:::

We see a rarefaction wave propagating away from the initial discontinuity, a contact discontinuity at x = 0.7, and a shock at x =0.9. 
