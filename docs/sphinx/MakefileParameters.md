(makefile-parameters)=
# Makefile Parameters

Most of the configuration options in Cholla currently require turning on or off flags in a make.type file. This page contains most of the available options, along with a brief summary of their meanings.

Developer Documentation about how these parameters are used and how to add a new parameter are provided on {ref}`this linked pag <dev-build-time-config>`.

## General options

* **DISABLE_GPU_ERROR_CHECKING**: Turns off optional GPU error checking. GPU error checking is on by default with a device sync that may impact performance. Passing this macro flag disables GPU error checking altogether and may improve performance.

* **MPI_CHOLLA**: Turns on MPI version of Cholla. Output files will be appended with the process number.

* **SET_MPI_GRID**: Set the MPI process grid decomposition manually. Can be useful for scaling tests, etc.

* **PRECISION** = 1 or 2 : Run with single or double precision. (Note, Cholla has not been tested with single precision recently.)
* **N_STEPS_LIMIT** = : Limit the total number of steps taken to a certain number (useful primarily for testing).

## Output options
* **OUTPUT**: Output data files. If neither BINARY or HDF5 are turned on, will default to text outputs (not recommend for multi-D problems).
* **BINARY**: Output binary files. No longer supported, but code exists.
* **HDF5**: Output hdf5 files. Requires the use of the hdf5 library when compiling.
* **N_OUTPUT_COMPLETE** = : Output full grid every N files. Can also be accomplished with the "nfull" parameter in the input file.
* **SLICES**: For 3D simulations, output x-z and y-z slices of density, x momentum, y momentum, z momentum, total energy, thermal energy, and passive scalars. Only compatible with HDF5 outputs.
* **PROJECTION**: For 3D simulations, output x-z and y-z projections of density and density-weighted temperature. Only compatible with HDF5 outputs.
* **ROTATED_PROJECTION**: For 3D simulations, output rotated projections of density and density-weighted temperature. Only compatible with HDF5 outputs.

## Spatial reconstruction methods
* **PCM**: Use piecewise constant reconstruction for hydro integrator. Very diffusive, useful primarily for testing.
* **PLMP**: Use piecewise linear reconstruction with limiting in the primitive variables. Follows the MUSCL scheme outlined in Toro to advance interface states in time (2006).
* **PLMC**: Use piecewise linear reconstruction with limiting in the characteristic variables. Follows the procedure outlined in Stone (2008).
* **PPMP**: Use piecewise parabolic reconstruction with limiting in the primitive variables. Follows the procedure outlined in Fryxell (2000). Contact discontinuity steeping and shock flattening are turned off by default, but can be turned on in the "ppmp_cuda.cu" source file.
* **PPMC**: Use piecewise parabolic reconstruction with limiting in the characteristic variables. Follows the procedure outlined in Stone (2008).

## Riemann Solvers
* **EXACT**: Use an exact Riemann solver, as described in Toro (2006).
* **ROE**: Use a Roe Riemann solver, as described in Stone (2008).
* **HLLC**: Use an HLLC Riemann solver, as described in Stone (2008).
* **HLLD**: Use an HLLD Riemann solver, as described in [Miyoshi & Kusano (2008)](https://www.sciencedirect.com/science/article/pii/S0021999105001142?via%3Dihub). This is the only solver that supports MHD at this time.

## Hydro Integrators
* **VL**: Use the Van Leer predictor-corrector method, as described in Stone (2009)
* **SIMPLE**: Use a simple dimensionally-split PLM or PPM style integrator, as described in Schneider (2017)

## Cooling functions
* **COOLING_GRACKLE**:

## Self Gravity
* **GRAVITY**: Turns on the gravity solver for gas self-gravity; requires choosing a solver (see below)
* **GRAVITY_GPU**: Gravity arrays reside in GPU memory only (are not copied over to the CPU)

One of the following two solvers must be selected:
* **PARIS**: A block based 3D FFT solver
* **SOR**: A successive over relaxation solver (currently not supported with GRAVITY_GPU)

Optional parameters:
* **OUTPUT_POTENTIAL**: Write out the potential to file
* **GRAVITY_5_POINTS_GRADIENT**: Use a 5 point approximation when estimating the potential at the cell center (by default 3 point is used)

## Particle Gravity
Note: Particles can be used alone, but in order to use particles + hydro, FFT Gravity must also be turned on
* **PARTICLES**: Turn on particle solver
* **PARTICLES_CPU**: Use the CPU to evolve the particles
* **PARTICLES_GPU**: Use the GPU to evolve the particles
* **ONLY_PARTICLES**: Run an n-body simulation only (no hydro)
* **SINGLE_PARTICLE_MASS**: Set a single mass for all the particles (if not selected, particle masses must be set in initial conditions)
* **PARTICLE_IDS**: Assign IDs to particles
* **PARTICLE_AGE**: Assign times to particles

## Feedback
Note: feedback requires PARTICLES, PARTICLES_GPU, PARTICLE_IDS, PARTICLE_AGE, DE
* **FEEDBACK**: Turn on feedback
* **ONLY_RESOLVED**: Treat all supernovae as though shell formation is resolved
* **NO_WIND_FEEDBACK**: Turn off wind feedback
* **NO_SN_FEEDBACK**: Turn off supernova feedback

## Additional options
* **DE**: Use the dual energy formalism to advect / correct estimates for internal energy. More details are provided [here](https://github.com/cholla-hydro/cholla/wiki/Dual-Energy-Formalism).

* **SCALAR**: Initialize additional scalar fields. Must be used with at least one additional build flag, such as `BASIC_SCALAR`, `DUST`, or `CHEMISTRY_GPU`. New additional scalar fields can also be added according to [these instructions](https://github.com/cholla-hydro/cholla/wiki/Grid_Enum-and-adding-passive-scalars).

* **BASIC_SCALAR**: Initialize an unused scalar field. See info on accessing the `basic_scalar` field [here](https://github.com/cholla-hydro/cholla/wiki/Grid_Enum-and-adding-passive-scalars).

* **STATIC_GRAV**: Turn on static gravity. Currently, gravitational forces are added by hand by modifying the relevant functions in "gravity_cuda.cu".

* **DENSITY_FLOOR**: Apply a density floor to all cells. The specific numerical value can be changed by specifying the `density_floor` variable in the input parameter file in $\textrm{g}/\textrm{cm}^3$. The default value is zero.

* **TEMPERATURE_FLOOR**: Apply a temperature floor to all cells. The specific numerical value can be changed by specifying the `temperature_floor` variable in the input parameter file in $\textrm{K}$. The default value is zero.

* **SCALAR_FLOOR**: Apply a scalar floor to all cells. The particular scalar the floor is applied to is specified using the `field_num` argument of `Apply_Scalar_Floor` when it is called in [src/grid/grid3D.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/grid/grid3D.cpp), and should be specified with its `grid_enum` index (see [grid_enum documentation](https://github.com/cholla-hydro/cholla/wiki/Grid_Enum-and-adding-passive-scalars) for more info). The default value is zero.

* **AVERAGE_SLOW_CELLS**: If a cell is causing the timestep to become very small relative to a set number, average the conserved quantities for that cell with its nearest neighbors. This requires "min_dt_slow" to be set, which is done by default only when GRAVITY is turned on.

* **CPU_TIME**: Measure / print out some timing statistics for a simulation as it runs. Also prints out the GPU memory usage statistic on every time step

* **MHD**: Enable magnetic fields. Only works with the HLLD solver and the Van Leer integrator and only in 3D

* **DUST**: Initialize a dust density field. Requires that the `SCALAR` flag is also turned on.

* **METALS**: Initialize a metal density field. Requires that the `SCALAR` flag is also turned on.
