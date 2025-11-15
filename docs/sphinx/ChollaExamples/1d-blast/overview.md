# 1D Blast Wave
This test is designed to assess the performance of a code near strong shocks and contact discontinuities. Parameters are derived from Woodward & Collela, 1984. The test consists of three regions. For x < 0.1, pressure is set to 1000.0. For x > 0.9, pressure is set to 100. Everywhere else, pressure is set to 0.01. Density is set to 1.0 everywhere and velocity everywhere is zero. Gamma is set to 1.4. This test is performed with the hydro build (`cholla/builds/make.type.hydro`) and Van Leer integrator. Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Blast_1D()`. 

## Parameter file: (`cholla/examples/1D/blast_1D.txt`)
```
#
# Parameter File for the 1D interacting blast wave test from
#     Woodward & Collela, 1984. See also Stone et al., 2008, Section 8.1
#

######################################
# number of grid cells in the x dimension
nx=400
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=0.038
# time interval for output
outstep=0.00038
# value of gamma
gamma=1.4
# name of initial conditions
init=Blast_1D
# domain properties
xmin=0.0
ymin=0.0
zmin=0.0
xlen=1.0
ylen=1.0
zlen=1.0
# type of boundary conditions
xl_bcnd=2
xu_bcnd=2
yl_bcnd=0
yu_bcnd=0
zl_bcnd=0
zu_bcnd=0
# path to output directory
outdir=./
```
Upon completion, you should obtain 101 output files. The initial and final density, pressure, and velocity (in code units) of the solution is shown below.  Examples of how to extract and plot data can be found in `cholla/python_scripts/plot-1d-blast.py`.  

We see a contact discontinuity around x = 0.6 followed by a shock at x = 0.65 and a rarefaction fan. We have a contact discontinuity just past x = 0.75 cells and a shock at x = 0.85. 

:::{figure} blast.png 
:::

We can also obtain the evolution:

:::{video} docs-blast.mp4
    :width: 700
    :height: 500
    :align: center
    :autoplay:
    :loop:
:::
