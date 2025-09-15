# Cooling/Chemistry

Cholla includes several equilibrium cooling and heating functions that act on the gas in a hydrodynamic simulation.

Historically, this choice of solver could **ONLY** be configured at compile-time.
Now, a number of options can be selected at runtime.

The choice of solver is controlled by the {par:param}`chemistry.kind` parameter.
For backwards compatibility, we set the default based on the presence of Makefile parameters.
When no Makefile parameters are provided, the parameter defaults to "none."

With the exception of "chemistry-gpu" and "grackle" (more on those in a moment), you can freely overwrite the default value at runtime.
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
* - "chemistry-gpu"
  - Non-equilibrium Hydrogen/Helium chemical network with heating/cooling
  - `CHEMISTRY_GPU`
* - "grackle"
  - CPU-based non-equilibrium primordial chemistry with heating/cooling
  - `COOLING_GRACKLE`
:::

:::{important}
At the time of writing, if you want to use "chemistry-gpu," you *MUST* use the `CHEMISTRY_GPU` Makefile parameter.
Likewise, if you want to use "grackle", you *MUST* use the `COOLING_GRACKLE` Makefile parameter.
In both cases, we do not allow the choices to be overwritten.
:::


## Rework me 
These include: a piecewise-parabolic fit to a collisional ionization equilibrium (CIE) cooling function at solar metallicity (see Appendix A.2 in [Schneider & Robertson, 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S/abstract) for a description); a cooling / heating function based on a solar metallicity Cloudy model with a Hardt & Madau 2005 UV background (see Appendix C in [Schneider & Robertson, 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...834..144S/abstract) for a description), and H/He chemistry/cooling based on the Grackle library with a uniform UVB background for photoheating and photoionization.
Each of these cooling and heating options are turned on via compile-time macros, listed below.

## Flags

The `COOLING_GPU` flag activates gpu-native cooling, which is either the default CIE cooling or Cloudy Cooling. 

The `CLOUDY_COOL` flag overrides default CIE cooling with Cloudy Cooling from cooling tables. 

The `CHEMISTRY_GPU` flag turns on non-equilibrium H+He chemistry and cooling

The `COOLING_GRACKLE` flag activates CPU-native Grackle cooling. (deprecated but code still exists)

## CIE Cooling

CIE cool provides an analytic fit to a solar metallicity CIE cooling curve calculated using Cloudy, with no cooling below 1e4 K.

## Cloudy Cooling 

Cloudy cooling loads data from relative paths ./cloudy\_coolingcurve.txt or src/cooling/cloudy\_coolingcurve.txt (see {repository-file}`src/cooling/load_cloudy_texture.cu`).
A copy of the Cloudy table can be found at {repository-file}`src/cooling/cloudy_coolingcurve.txt`.

The provided table spans densities `log(n) = -6` to `log(n) = 6` (cm^-3) and temperatures from `log(T) = 1` to `log(T) = 9` (K) with no cooling for temperatures below 10 K.

Cooling is computed from the table using bilinear interpolation.
A custom function is used for double precision because the CUDA built-in performs interpolation using 8-bit accuracy.

## H/He Chemistry/Cooling

Tracks non-equilibrium ionization states of hydrogen and helium and calculates cooling/heating rates based on a uniform UV background.
Time-dependent UVB phothoheating and photoionization rates are passed as a text file, with an input parameter specifying the text file name.

## Parametters

% todo: with some minimal changes, we can autogenerate this table (or provide the
%       option to autogenerate part of the table)

:::{list-table}
:widths: auto
:header-rows: 1

* - Name
  - Summary
* - {par:param}`chemistry.kind`
  - kind of chemistry solver to use
* - {par:param}`chemistry.data_file`
  - path to data file used by "tabulated-cooling" solver
:::
