# Cooling/Chemistry

Cholla includes several equilibrium cooling and heating functions that act on the gas in a hydrodynamic simulation.


## Available Solvers

Cholla currently offers 2 flavors of solvers:

1. Simplified solvers that just model the effect of cooling/heating on the thermal energy of gas.

2. More complex solvers that solve non-equilibrium primordial chemistry (i.e. H/He) and cooling.

Historically, this choice of solver could **ONLY** be configured at compile-time.
Now, members of the first flavor are specified through the {par:param}`chemistry.kind` runtime parameter (the other flavor still needs to be chosen at compile time).
We will cover this in more detail down below.

## Simple Cooling Solvers

Supported options include:
- `"piecewise-cie"`: a piecewise-parabolic fit to a collisional ionization equilibrium (CIE) cooling function at solar metallicity (see Appendix A.2 in [Schneider & Robertson, 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S/abstract) for a description)
- `"piecewise-ti"`: Analytic cooling/heating recipe that roughly matches the "TI" cooling runs shown in
 [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
- `"tabulated-cloudy"`: a cooling / heating function based on a solar metallicity Cloudy model with a Hardt & Madau 2005 UV background (see Appendix C in [Schneider & Robertson, 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...834..144S/abstract) for a description)

### Selecting your solver

The choice of simple-solver is specified through the {par:param}`chemistry.kind` parameter.
For backwards compatibility, we set the default based on the presence of Makefile parameters.
When no Makefile parameters are provided, the parameter defaults to "none."
The following table summarizes the available choices (and the conditions where they become defaults):

:::{list-table}
:widths: auto
:header-rows: 1

* - Name
  - Meaning
  - Macros that make this the default
* - "none"
  - No chemistry or cooling
  - N/A
* - "tabulated-cloudy"
  - path to data file used by "tabulated-cooling" solver
  - `COOLING_GPU` && `CLOUDY_COOL`
* - "piecewise-cie"
  - piecewise-parabolic fit to a collisional ionization equilibrium (CIE)
  - `COOLING_GPU` **without** `CLOUDY_COOL`
* - "piecewise-ti"
  - Analytic cooling/heating recipe that roughly matches the "TI" cooling runs shown in
 in [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
  - N/A
:::

:::{important}
We are planning to phase out the `COOLING_GPU` && `CLOUDY_COOL` Makefile parameters.
:::

### More about the solvers

#### "piecewise-cie"

This solver provides an analytic fit to a solar metallicity CIE cooling curve calculated using Cloudy, with no cooling below 1e4 K.

Not currently compatible with photoelectric heating.

#### "piecewise-ti"

This solver uses the same analytic fit as "piecewise-cie" above 1e4K and extends the fit at lower temperatures.

**Compatible with photoelectric heating.**

#### "tabulated-cloudy"

Cloudy cooling loads data from relative paths ./cloudy\_coolingcurve.txt or src/cooling/cloudy\_coolingcurve.txt (see {repository-file}`src/cooling/load_cloudy_texture.cu`), or the path specified via the {par:param}`chemistry.data_file` parameter.
A copy of the Cloudy table can be found at {repository-file}`src/cooling/cloudy_coolingcurve.txt`.

The provided table spans densities `log(n) = -6` to `log(n) = 6` (cm^-3) and temperatures from `log(T) = 1` to `log(T) = 9` (K) with no cooling for temperatures below 10 K.

Cooling is computed from the table using bilinear interpolation.
A custom function is used for double precision because the CUDA built-in performs interpolation using 8-bit accuracy.

**Compatible with photoelectric heating.**

### Modelling Photoelectric Heating

Several of our solvers are described above can be configured so that they also model the effect of photoelectric heating.
To enable photoelectric heating, you can use the {par:param}`chemistry.photoelectric_heating` option.
When photoelectric heating is enabled, you can control the assumed average number density with the {par:param}`chemistry.photoelectric_n_av_cgs` parameter.

:::{todo}
It would be great to show the actual equations that get used
:::

## Chemistry and Cooling Solvers

These solvers are based on the [Grackle library](https://grackle.readthedocs.io/en/latest/) with and model a uniform time-dependent UVB background for photoheating and photoionization.
When these solvers are used the primordial species (e.g. HI, HII, HeI, HeII, HeIII, electrons) are all modelled as passive scalars.

There are currently 2 choices of solvers:
- `"grackle"`: it directly uses the Grackle library. At the time of writing:

  - this can only run on CPUs (but efforts are ongoing to directly add GPU support to the Grackle library).

  - this is hardcoded to use Grackle's `primordial_chemistry=1` chemical network

- `"chemistry-gpu"`: uses logic implemented directly as part of Cholla to run chemistry on GPUs (the logic was closely modelled after chemisry

Time-dependent UVB phothoheating and photoionization rates are passed as a text file, with an input parameter specifying the text file name.

### Enabling the solver

:::{important}
At the time of writing, if you want to use "chemistry-gpu" or the "grackle" solvers you *MUST* use the Makefile flags.

The `CHEMISTRY_GPU` flag turns on non-equilibrium H+He chemistry and cooling

The `COOLING_GRACKLE` flag activates CPU-native Grackle cooling. (deprecated but code still exists)
:::

