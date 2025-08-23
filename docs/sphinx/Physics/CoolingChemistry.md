# Cooling/Chemistry

:::{todo}
Is this up to date? There have been a number of recent changes and this is now somewhat outdated
:::

Cholla includes several equilibrium cooling and heating functions that act on the gas in a hydrodynamic simulation.
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

