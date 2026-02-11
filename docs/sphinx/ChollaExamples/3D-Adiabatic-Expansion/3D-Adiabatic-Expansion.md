# 3D Adaibatic Expansion Test

This test demonstrates simple adiabatic expansion in a cosmological frame.  The initial conditions are the critical density in baryons at the starting redshift, T=100K, and no bulk velocity. Gamma is set to 5/3. This test is performed with the cosmology build (`cholla/builds/make.type.cosmology`) and SIMPLE integrator.
Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Adiabatic_Expansion()`. 

## Parameter file:

This parameter file can be found in [examples/3D/Adiabatic_Expansion.txt](https://github.com/cholla-hydro/cholla/blob/main/examples/3D/Adiabatic_Expansion.txt) on the `dev` branch.
```
#
# Parameter File for the 3D Adiabatic Expansion test.
#

######################################
# number of grid cells in the x dimension
nx=256
# number of grid cells in the y dimension
ny=32
# number of grid cells in the z dimension
nz=32
# output time
tout=1000
# how often to output
outstep=1000
# value of gamma
gamma=1.66666667
# name of initial conditions
init=Adiabatic_Expansion
#Cosmological Parameters
Init_redshift=20.0
#Init_redshift=0.998294693667
H0=50.0
Omega_M=1.0
Omega_L=0.0
Omega_b=1.0
temperature_floor=1.0e-2
scale_outputs_file=scale_output_files/outputs_adiabatic_expansion.txt
# domain properties
xmin=0.0
ymin=0.0
zmin=0.0
xlen=64000.0
ylen=8000.0
zlen=8000.0
# type of boundary conditions
xl_bcnd=1
xu_bcnd=1
yl_bcnd=1
yu_bcnd=1
zl_bcnd=1
zu_bcnd=1
# path to output directory
indir=ics/
outdir=./
```
  
Upon completion, you should obtain two output files. The final density, velocity, and temperature in physical units are shown below. 
:::{figure} adiabatic_expansion.png
Adiabatic Expansion test solution from Cholla.
::: 