# 1D Two Shocks
This test highlights a collision between two shocks. This test is from Toro's *Riemann solvers and numerical methods for fluid dynamics* Sec. 6.4, test 4.The test consists of left and right states separated at x = 0.4 with velocities 19.5975 and -6.19633, respectively. Density of the left side is 5.99924 and pressure is 460.894. For the right side, density is 5.99242 while pressure is 46.095. Gamma is set to 1.4. This test is performed with the hydro build (`cholla/builds/make.type.hydro`). Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Riemann()`.  

## Parameter file: (**modified** to add y and z boundary conditions = 0 from `cholla/examples/1D/two_shocks.txt`)
```
#
# Parameter File for Toro test 4, a collision of two shocks.
# Parameters derived from Toro, Sec. 6.4.4, test 4
#

################################################
# number of grid cells in the x dimension
nx=100
# number of grid cells in the y dimension
ny=1
# number of grid cells in the z dimension
nz=1
# final output time
tout=0.035
# time interval for output
outstep=0.035
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
rho_l=5.99924
# velocity of left state
vx_l=19.5975
vy_l=0.0
vz_l=0.0

# pressure of left state
P_l=460.894
# density of right state
rho_r=5.99242
# velocity of right state
vx_r=-6.19633
vy_r=0.0
vz_r=0.0
# pressure of right state
P_r=46.095
# location of initial discontinuity
diaph=0.4
# value of gamma
gamma=1.4
```
Upon completion, you should obtain 2 output files. The initial and final density, pressure, velocity, and internal energy of the solution is shown below. Examples of how to extract and plot data can be found in `cholla/python_scripts/plot_sod.ipynb`.  

:::{figure} intial-twoshocks.png

:::{figure} final-twoshocks.png

<img src="./images/1d_two_shocks_6panel_density_pressure.png" alt="Three rows of two scatter plots side by side. The first row shows density vs x position, the second row shows pressure vs x position, and the third row shows velocity vs position. In all rows, the first plot has the text 't = 0.00' in the upper right corner while the second plot has the text 't = 0.04' in the upper right corner. The plots of the first column are shown with pink dots while the plots of the second column have pink dots plotted over a purple line. In all cases, the pink dots match the shape of the purple line, very closely for the velocity plot and imperfectly for the pressure and density. All plots range from 0.0 to 1.0 on the x-axis. The initial density plot shows a value of approximately 6 for all x. The final density plot shows a value of 5.99924 for x = 0 to x = 0.4. It then jumps discontinuously to a value of 15. It jumps again, albeit less sharply, to a value of 33 at x = 0.7. It remains here until x = 0.8 where is drops discontinuously to a value of 5.99924, where it stays for the last 0.2 in x. The initial pressure plot has a value of 460.894 between x = 0 and 0.4 and 46.095 elsewhere. The final pressure plot has a constant value of 460.894 from x = 0 to x= 0.4 before jumping discontinuously to a value of 1700. It drops to a value of 50 at x = 0.8 and remains there for the last 0.2 in x. The initial velocity plot has a value of approximately 20 for x less than 0.4 and approximately -6 for x greater than 0.4. The final velocity plot shows a value of 20 until x = 0.4, where it drops to a value of 10. It remains at 10 until x = 0.8, where it drops again to -6." width="1200" />  

We see a shock, contact discontinuity, and a shock, which is in agreement with Toro's solution and can be seen below:  

<img src="./images/toro2013test4.png" width="800" />  

