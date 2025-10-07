# 1D Strong Shock
This test is similar to the Sod shock tube but has higher initial pressure and density differences. This shows the ability of a code to limit oscillatory behavior in areas of high density and pressure contrasts. The setup consists of a density and pressure of 10.0 and 100.0, respectively, for 0 \< x \< 0.5 and density= pressure = 1.0 for 0.5 \< x \< 1.0. Gamma is set to 1.4. This test was performed with the hydro build (`cholla/builds/make.type.hydro`) and Van Leer integrator. Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Riemann()`. 

## Parameter file: (modified from`cholla/examples/1D/strong_shock.txt`)
Modified to add yl_bcnd, yu_bcnd, zl_bcnd, and zu_bcnd=0
```
#
# Parameter File for 1D strong shock test
#

################################################
# number of grid cells in the x dimension
nx=100
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=0.07
# time interval for output
outstep=0.07
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
rho_l=10.0
# velocity of left state
vx_l=0.0
vy_l=0.0
vz_l=0.0
# pressure of left state
P_l=100.0
# density of right state
rho_r=1.0
# velocity of right state
vx_r=0.0
vy_r=0.0
vz_r=0.0
# pressure of right state
P_r=1.0
# location of initial discontinuity
diaph=0.5
# value of gamma
gamma=1.4
```
Upon completion, you should obtain two output files.  The initial and final density, pressure, and velocity (in code units) of the solution is shown below (pink dots) plotted over the exact solution (purple line). Examples of how to extract and plot data can be found in cholla/python_scripts/plot_sod.ipynb.  

:::figure{two_times.png}

We see a rarefaction expanding from just after the initial discontinuity, followed by a contact discontinuity at x =0.75 and a shock at x = 0.85. There is very slight oscillatory behavior around x = 0.7 but it is limited.
