# 2D Rayleigh-Taylor Test

This test demonstrates the mixing resulting from a dense fluid placed on top of a less dense fluid. The bottom half of the grid is given of density of 1.0 while the top has a value of 2.0. Y velocities across the grid are set as a small pertubation tapering off from the center. For both halves, pressure is initialized as decreasing with increasing y position. Gamma is set to 1.4. This test is performed with the static gravity build ({repository-file}`config/make.type.static_grav`) and Van Leer integrator.
Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Rayleigh_Taylor()`. 

The parameter file can be found at: {repository-file}`examples/2D/Rayleigh_Taylor.txt`

## Parameter file:

Parameter file can be found on the `dev` branch.
```
#
# Parameter File for the 2D Rayleigh-Taylor test.
#

######################################
# number of grid cells in the x dimension
nx=200
# number of grid cells in the y dimension
ny=400
# number of grid cells in the z dimension
nz=1
# final output time
tout=5.0
# time interval for output
outstep=0.05
# value of gamma
gamma=1.4
# name of initial conditions
init=Rayleigh_Taylor
#static gravity flag
custom_grav=2
# domain properties
xmin=0.0
ymin=0.0
zmin=0.0
xlen=0.33333333
ylen=1.0
zlen=1.0
# type of boundary conditions
xl_bcnd=1
xu_bcnd=1
yl_bcnd=2
yu_bcnd=2
zl_bcnd=0
zu_bcnd=0
# path to output directory
outdir=./
```
To run on main:
You must add the following lines to `src/gravity/static_grav.h` under the function `inline __device__ void calc_g_2D()`:
```
*gx = 0;
*gy = -1;
```
Any other values assigned to `*gx` and `*gy` should be commented out.  
  
Upon completion, you should obtain 101 output files. The initial, intermediate, and final density and pressure (in code units) is shown below. Examples of how to plot projections and slices can be found in the [General 2D plotting example](../../PythonExamples/2D-plotting.md).
:::{figure} rayleigh_taylor_2D_xy.png
Rayleigh-Taylor test solution from Cholla.
::: 

This is comparable to the solution from Liska & Wendroff (2003):
:::{figure} liska203-rayleigh-taylor.png
Rayleigh-Taylor test solution from Liska & Wendroff (2003).
:::
