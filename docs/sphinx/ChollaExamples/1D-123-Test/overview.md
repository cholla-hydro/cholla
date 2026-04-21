# 1D Einfeldt Strong Rarefaction (123 test)

This test highlights a scenario in which linearized Riemann solvers will produce negative densities/pressures without appropriate modifications. This is due to a region of mostly kinetic energy and approaching vacuum pressure. 
However, `cholla` checks densities/pressure produced by the approximate (Roe) solver and switches to the HLLE solver if needed.
This is not an issue with the exact solver. 

This test is from Toro's [*Riemann solvers and numerical methods for fluid dynamics*](https://link.springer.com/book/10.1007/b79761) Sec. 6.3.3, test 2.
The test consists of left and right states separated at x = 0.5 with velocities -2 and 2, respectively, moving away from each other. Density = 1.0 and pressure = 0.4 on both sides. Gamma is set to 1.4. 
This test is performed with the hydro build ({repository-file}`builds/make.type.hydro`) and Van Leer integrator. 
Full initial conditions can be found in {repository-file}`src/grid/initial_conditions.cpp` under `Riemann()`.

## Parameter file:

:::{literalinclude} input.txt
:::

## Result

Upon completion, you should obtain 2 output files.
The initial and final density, pressure, and velocity (in code units) of the solution is shown below (pink dots) plotted over the exact solution (purple line).
Examples of how to extract and plot data can be found in the [General 1D Plotting Example](../../PythonExamples/1D-plotting.md).  .

:::{figure} 1d123-6panel-density-pressure.png
:::

We see that the two rarefaction waves have moved outwards, away from each other, leaving behind a region of low density and low pressure. 
