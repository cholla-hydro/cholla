# 2D Disk
This models the Milky Way's stellar disk following a Kuzmin profile. Gamma is set to 1.001. Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Disk_2D()`. This test is performed with the static gravity build (`cholla/builds/make.type.static_grav`) and Van Leer integrator.  

## Parameter file: (`cholla/examples/2D/disk.txt`): dev branch
```
#
# Parameter File for a 2D disk in keplerian rotation.
#

######################################
# number of grid cells in the x dimension
nx=512
# number of grid cells in the y dimension
ny=512
# number of grid cells in the z dimension
nz=1
# final output time
tout=1092950
# time interval for output
outstep=2185.9
# value of gamma
gamma=1.001
# name of initial conditions
init=Disk_2D
# static gravity flag
custom_grav=4
# domain properties
xmin=-20
ymin=-20
zmin=-20
xlen=40
ylen=40
zlen=40
# type of boundary conditions
xl_bcnd=3
xu_bcnd=3
yl_bcnd=3
yu_bcnd=3
zl_bcnd=3
zu_bcnd=3
# path to output directory
outdir=./
```
While the parameter file claims to be for a 2D disk in keplerian rotation (custom_grav = 3), the initial condition Disk_2D matches the ICs of custom_grav = 4.   
  
Upon completion, you should obtain 501 output files. (Note: when running with the Simple integrator, there is a thread crash. However, the Van Leer integrator is able to run the simulation to completion.) An evolution of the density is shown below. Examples of how to plot projections and slices can be found in `cholla/python_scripts/Projection_Slice_Tutorial.ipynb`.  

:::{video} disk-docs.mp4
    :width: 700
    :height: 500
    :align: center
    :autoplay:
    :loop:
:::