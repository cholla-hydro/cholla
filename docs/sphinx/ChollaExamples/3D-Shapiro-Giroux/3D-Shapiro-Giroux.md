(3D-Shapiro-Giroux)=
# 3D Shapiro-Giroux Cosmological Strömgren Sphere Test

This test follows the growth of an H II region around a point source in an expanding universe.
It is the cosmological generalization of the classical Strömgren sphere, and of Test 1 of [Iliev et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1057I/abstract), with the analytic solution of [Shapiro & Giroux (1987)](https://ui.adsabs.harvard.edu/abs/1987ApJ...321L.107S/abstract).
It exercises M1 radiative transfer coupled to the GPU chemistry in comoving coordinates (`RT_M1` + `COSMOLOGY` + `CHEMISTRY_GPU`); see {doc}`../../Physics/RT`.

The test uses the cosmological M1 build ({repository-file}`config/make.type.cosmo-m1`) with the SIMPLE integrator.
To also write and check the gas temperature, add `DFLAGS += -DOUTPUT_TEMPERATURE` to that file.
The initial conditions are in `cholla/src/grid/initial_conditions.cpp` under `Shapiro_Giroux_Cosmo()`.

## Setup

- An Einstein–de Sitter universe: {math}`\Omega_M = 1`, {math}`\Omega_\Lambda = 0`, {math}`H_0 = 70` km/s/Mpc.
- Uniform, static, pure-hydrogen gas ({math}`Y_\mathrm{He} = 0`) with {math}`\Omega_b = 0.1`. This gives a comoving hydrogen density {math}`\bar{n}_\mathrm{H} = \Omega_b \rho_\mathrm{crit} / m_p = 5.50 \times 10^{-7}` cm{math}`^{-3}`, and a proper density of {math}`2.82 \times 10^{-4}` cm{math}`^{-3}` at the starting redshift.
- The gas starts fully neutral at {math}`T_\mathrm{init} = 10^4` K, and Case B recombination is used.
- A point source at the box center switches on at {math}`z_i = 7`. It emits {math}`\dot{N}_\gamma = 5 \times 10^{48}` photons/s at the H I ionization threshold (the Iliev Test 1 spectrum), spread over the eight cells nearest the center.
- The run continues to {math}`z = 0` in a 2 comoving Mpc/{math}`h` box with {math}`128^3` cells.

## Analytic solution

Shapiro & Giroux (1987) normalize the scale factor to 1 at source turn-on.
In terms of the usual scale factor {math}`a` (with {math}`a = 1` today), {math}`a_i = 1/(1+z_i)`, and the comoving density {math}`\bar{n}_\mathrm{H}`, define

```{math}
K = \left[\frac{3\,\dot{N}_\gamma}{4\pi\,\alpha_B\,\bar{n}_\mathrm{H}^{\,2}}\right]^{1/3}.
```

The comoving Strömgren radius then grows with the universe as {math}`R_S(a) = a\,K` (their eq. 3).
The ionization front lags behind it.
Its position {math}`y = (r_I / r_{S,i})^3` obeys

```{math}
\frac{dy}{dx} = \lambda \left(1 - \frac{y}{(a/a_i)^3}\right), \qquad x = t/t_i, \qquad \lambda = \alpha_B\, n_{\mathrm{H},i}\, t_i ,
```

where {math}`n_{\mathrm{H},i} = \bar{n}_\mathrm{H}/a_i^3` is the proper density at turn-on and {math}`t_i` is the age of the universe at turn-on (their eq. 5).
For an Einstein–de Sitter universe the solution is (their eq. 10a)

```{math}
y(t) = \lambda\, \exp\!\left(\frac{\lambda t_i}{t}\right) \left[\frac{t}{t_i}\, E_2\!\left(\frac{\lambda t_i}{t}\right) - E_2(\lambda)\right],
```

with {math}`E_2` the exponential integral of order 2.
The comoving front radius is {math}`R_I(a) = a_i\, K\, y^{1/3}`.

For the parameters above with {math}`\alpha_B = 2.6 \times 10^{-13}` cm{math}`^3` s{math}`^{-1}`:

- {math}`t_i = 411.8` Myr and {math}`\lambda = 0.951`;
- the proper Strömgren radius at turn-on is 12.5 kpc;
- {math}`K = 802.1` comoving kpc, i.e. 561.5 comoving kpc/{math}`h`, so {math}`R_S(a = 1)` is 561.5 kpc/{math}`h`;
- the analytic front reaches {math}`R_I = 186.9` comoving kpc/{math}`h` at {math}`z = 0`.

The analysis script also integrates the front equation numerically, which works for any cosmology.
It checks the result against eq. (10a) when the cosmology is Einstein–de Sitter.

## Parameter file:

This parameter file can be found in {repository-file}`examples/3D/iliev/shapiro_giroux_cosmo.txt` on the `dev` branch.
Snapshot scale factors are listed in {repository-file}`scale_output_files/outputs_shapiro_giroux_cosmo.txt`.

```toml
######################################
# Parameter file for the Shapiro_Giroux_Cosmo test:
# cosmological Stromgren sphere (Shapiro & Giroux 1987)
######################################
# number of grid cells in the x dimension
nx=128
# number of grid cells in the y dimension
ny=128
# number of grid cells in the z dimension
nz=128
# output time
tout=1000
# how often to output
outstep=1000
# value of gamma
gamma=1.66666667
# name of initial conditions
init="Shapiro_Giroux_Cosmo"
# RT: maximum number of substeps per hydro step
rt_num_iterations=30
# RT: reduced speed of light; cosmological runs need rt_reduced_c > 0
rt_reduced_c=1.0
# snapshot scale factors
scale_outputs_file="scale_output_files/outputs_shapiro_giroux_cosmo.txt"
# domain properties (comoving kpc/h)
xmin=0.0
ymin=0.0
zmin=0.0
xlen=2000.0
ylen=2000.0
zlen=2000.0
# type of boundary conditions
xl_bcnd=1
xu_bcnd=1
yl_bcnd=1
yu_bcnd=1
zl_bcnd=1
zu_bcnd=1
# path to output directory
outdir="data/"
# Hubble parameter in km/s/Mpc
H0=70.0
# matter, baryon, dark energy and radiation density parameters
Omega_M=1.0
Omega_b=0.1
Omega_L=0
Omega_R=0
# initial ionization fractions of H and He
xHp_ion_init=0
xHep_ion_init=0
# initial gas temperature (K)
T_init=1e4
# helium mass fraction
YHe=0
# initial redshift (source turn-on)
Init_redshift=7.0
```

## Running the test and the analysis

Build with `make TYPE=cosmo-m1` and run from the repository root:

```bash
mkdir -p data
./bin/cholla.cosmo-m1.<machine> examples/3D/iliev/shapiro_giroux_cosmo.txt
```

Two analysis scripts in {repository-dir}`examples/3D/iliev` compare a snapshot with the analytic solution.
Both need `shapiro_giroux_cosmo_common.py` from the same directory, and both read single-process output files.

```bash
python examples/3D/iliev/shapiro_giroux_cosmo_analysis.py data/<n>.h5 examples/3D/iliev/shapiro_giroux_cosmo.txt
python examples/3D/iliev/shapiro_giroux_cosmo_slices.py   data/<n>.h5 examples/3D/iliev/shapiro_giroux_cosmo.txt
```

`shapiro_giroux_cosmo_analysis.py` does the following:

- builds the spherically averaged ionized fraction {math}`x_\mathrm{HII}` about the source;
- reports two simulated front radii: {math}`R_\mathrm{thresh}`, where {math}`x_\mathrm{HII}` falls below 0.5, and the volume-equivalent radius {math}`R_\mathrm{vol} = (3 V_\mathrm{ion}/4\pi)^{1/3}`;
- compares both with {math}`R_I(a)` and {math}`R_S(a)`.

If the snapshot contains `temperature`, it also checks it:

- At the first snapshot the gas must be at {math}`T_\mathrm{init}`.
- Later, the neutral gas must follow adiabatic cooling, {math}`T = T_\mathrm{init}(a_i/a)^2` for {math}`\gamma = 5/3`, within a factor of 2.
- The script plots the radial temperature profile and reports the median temperature of the ionized gas.

`shapiro_giroux_cosmo_slices.py` plots x-y, y-z and z-x slices through the source, with the expected {math}`R_S(a)` and {math}`R_I(a)` overlaid.
By default the top row is the neutral hydrogen density {math}`n_\mathrm{HI}` and the bottom row is {math}`x_\mathrm{HII}`.
`--rows` selects other quantities (`nhi`, `xhii`, `nh`, `temperature`).

The scripts also print two consistency checks:

- the mean comoving hydrogen density in the file against {math}`\Omega_b \rho_\mathrm{crit}/m_p`, which tests the density convention of the initial conditions;
- for Einstein–de Sitter cosmologies, the analytic front from the numerical integration against eq. (10a).

## Results

The figures below show the final snapshot at {math}`z = 0` ({math}`t = 9319` Myr).

:::{figure} shapiro_giroux_cosmo_comparison.png
Comparison with the analytic solution at {math}`z = 0`.
Left: spherically averaged ionized fraction, with the expected front {math}`R_I` (red dashed), the expected Strömgren radius {math}`R_S(a)` (black dotted) and the simulated {math}`R_\mathrm{vol}` (blue dotted).
Center: analytic {math}`R_I` and {math}`R_S(a)` against redshift, with the simulated radii at {math}`z = 0`.
Right: spherically averaged gas temperature.
:::

The volume-equivalent front radius is {math}`R_\mathrm{vol} = 193.5` comoving kpc/{math}`h`, within 3.5% of the analytic {math}`R_I = 186.9` kpc/{math}`h`.
The {math}`x_\mathrm{HII} = 0.5` radius lies slightly inside it, and the front is smooth over roughly 100 kpc/{math}`h`.
As Shapiro & Giroux predict, the front stays far inside the growing Strömgren radius ({math}`R_S = 561.5` kpc/{math}`h` at {math}`z = 0`).
At low redshift the recombination time exceeds the age of the universe.

The neutral gas cools adiabatically from {math}`10^4` K to the expected 156 K at {math}`z = 0`.
The ionized gas is much colder than the {math}`10^4` K assumed for {math}`\alpha_B` in the analytic solution.
It peaks at about 600 K just inside the front and falls below 100 K near the source.
This is because the source emits only at the H I threshold, so each photoionization deposits almost no heat.
The comparison therefore tests photon conservation and front propagation in comoving coordinates, rather than the thermal state of the ionized gas.

:::{figure} shapiro_giroux_cosmo_slices.png
Slices through the source at {math}`z = 0`.
Top: neutral hydrogen density {math}`n_\mathrm{HI}` (physical, cm{math}`^{-3}`).
Bottom: ionized fraction {math}`x_\mathrm{HII}`.
The white dashed circle is the analytic front {math}`R_I(a)` and the red dotted circle the Strömgren radius {math}`R_S(a)`.
:::

The H II region is spherical and centered on the source in all three planes, and its edge follows the analytic front.
Inside it, {math}`n_\mathrm{HI}` falls by about three orders of magnitude toward the source, compared with the surrounding neutral gas.
