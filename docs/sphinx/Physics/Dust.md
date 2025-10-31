# Dust

## Turning on `DUST`
The `SCALAR` and `DUST` flags can be used in any build to add dust as an additional fluid component to a simulation (see also the `DUST` build type). Note that Cholla's dust model treats dust as a passive scalar, so both flags are required. The dust field in the conserved variable arrays can be accessed using `grid_enum::dust_density`.

## Initializing dust distribution
The initial distribution of dust can be set in the same manner as the other conserved variables in the initial conditions function (see, for example, the `Clouds` initial conditions function). This controls the spatial distribution and initial density of dust. The dust grain size should be set using the `grain_radius` keyword in the input file (in units of 0.1 micron). Invocations of the `DUST` macro should be wrapped in the `SCALAR` macro.

## Dust model
Currently, Cholla's dust model simulates the destruction of dust due to sputtering, which depends on the local gas density and temperature, as well as the dust grain size. The sputtering model can be found in Section 2.1 of [Richie et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...974...81R/abstract), based on the model from [Draine & Salpeter (1979)](https://ui.adsabs.harvard.edu/abs/1979ApJ...231...77D/abstract). For the dust dynamics, we assume that gas and dust are fully dynamically coupled.