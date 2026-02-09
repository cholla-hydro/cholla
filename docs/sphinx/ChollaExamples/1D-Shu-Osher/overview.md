# 1D Shu and Osher test
This test (Shu & Osher, 1989) highlights the ability of a code to resolve small scale smooth flow and shocks simultaneously. Further, it shows how lower resolution solutions can cut off some of the amplitude of maxima due to the slope limiters. Parameters are from Stone et al., 2008, Section 8.1. The test consists of left and right states separated at x = -0.8. On the left, density is set to 3.857143, pressure to 10.33333, and velocity to 2.629369. On the right, density is sinusoidally varying: $\rho(x)$ =  1.0 + 0.2 $\sin(5.0\pi x)$. Pressure is set to 1.0 and the velocity is 0.0. Gamma is set to 1.4. This test is performed with the hydro build (`cholla/builds/make.type.hydro`). Full initial conditions can be found in `cholla/src/grid/initial_conditions.cpp`under `Shu_Osher()`.  


## Parameter file: ({repository-file}`examples/1D/Shu_Osher.txt`)
Modified to add yl_bcnd, yu_bcnd, zl_bcnd, and zu_bcnd=0. Xmin changed to -1.0 and xlen to 2.0.

:::{literalinclude} parameter-file.txt
:::

Upon completion, you should obtain 2 output files. The initial and final density, pressure, and velocity (in code units) of the solution is shown below. Examples of how to extract and plot data can be found in `cholla/python_scripts/plot_sod.ipynb`.  

:::{figure} shu-osher.png


With the diode disabled, this solution does match that of Schneider and Robertson 2015 and Stone et al. 2008, shown below:

:::{figure} schneider-robertson-2015.png
:width: 500px
:align: center
:::


