# Radiative Transfer

The ability to solve radiative transfer (RT) problems in Cholla is under development.
Two schemes exist: an Optically-Thin Variable Eddington Tensor (OTVET) scheme and an M1 scheme.
The M1 scheme can be combined with cosmology (`RT_M1` + `COSMOLOGY`), as described below.

## Optically-Thin Variable Eddington Tensor (OTVET) Scheme (in dev)

Nick Gnedin implemented his OTVET radiative transfer scheme (see, e.g., [Gnedin & Abel 2001](https://ui.adsabs.harvard.edu/abs/2001NewA....6..437G/abstract)).
The OTVET scheme has not been tested with `COSMOLOGY`.

## M1 Scheme (in dev)

An M1 scheme for Cholla is under development, building from Nick Gnedin's M1 implementation in his Altair code.
The original M1 reference is [C. D. Levermore, 1984](https://ui.adsabs.harvard.edu/abs/1984JQSRT..31..149L/abstract).

### What is solved

For each frequency group the scheme evolves the photon number density {math}`N` and the photon flux {math}`\mathbf{F}`.
The radiation pressure tensor is closed with the M1 (Levermore) closure.
Four groups are carried: the H I, He I and He II ionizing bands, plus a total-intensity field, so each cell holds 16 RT fields (one density and three flux components per group).

Absorption couples the radiation to the gas through the H I, He I and He II photoionization cross sections.
The radiation field in turn sets the photoionization and photoheating rates used by the GPU chemistry network (`CHEMISTRY_GPU`).
These rates come from tables built from the source spectrum, so the RT always requires `CHEMISTRY_GPU`.

### Time integration

The RT is operator split from the hydrodynamics and is advanced once per hydro step, before the hydro and chemistry update.
Within a hydro step the radiation is advanced in up to {par:param}`rt_num_iterations` substeps of a semi-implicit scheme.
In each substep, absorption in a cell is integrated exactly over the substep, which keeps the update stable in optically thick cells.

How far light travels per hydro step is set by {par:param}`rt_reduced_c`:

- `rt_reduced_c = 0` (default): the legacy scheme. Light advances one cell per hydro step, whatever the hydro time step, so the effective speed of light is {math}`\Delta x / \Delta t`. This is the setting the non-cosmological Iliev tests are validated with.
- `rt_reduced_c > 0`: a reduced speed of light {math}`c_\mathrm{eff} = f\,c` with {math}`f` = `rt_reduced_c`. The number of substeps follows from {math}`c_\mathrm{eff}\,\Delta t / \Delta x`. It is capped at `rt_num_iterations`, and the substep size is capped at the RT CFL number (0.5), so light can travel at most {math}`0.5 \times` `rt_num_iterations` cells per hydro step.

Each step the code prints the effective speed of light as `RT: c_eff/c = ...`, which shows whether the cap is active.

### Building and parameters

Two build types enable the M1 scheme:

- {repository-file}`config/make.type.m1`: hydrodynamics with `RT_M1` and `CHEMISTRY_GPU`.
- {repository-file}`config/make.type.cosmo-m1`: the same on top of the cosmology build ({repository-file}`config/make.type.cosmology`).

Both also enable `OUTPUT_CHEMISTRY` and `OUTPUT_RADIATION`.
Add `-DOUTPUT_TEMPERATURE` to write the gas temperature as well.

RT parameters (see the {doc}`parameter reference <../Reference/Parameters>` for details):

| Parameter | Default | Meaning |
|---|---|---|
| {par:param}`rt_num_iterations` | 10 | Maximum number of RT substeps per hydro step. |
| {par:param}`rt_reduced_c` | 0 | Reduced speed of light factor. 0 selects the legacy one-cell-per-step scheme; a value > 0 sets {math}`c_\mathrm{eff} = ` `rt_reduced_c` {math}`\times\, c`. |
| {par:param}`rt_outputs_file` | none | Optional list of RT output times. Used only without `COSMOLOGY`. |

### Outputs

With `OUTPUT_RADIATION`, each snapshot contains the source field `rf_sources` and, for each group `X` in `HI`, `HeI`, `HeII` and `0`, the datasets `rf_X` (photon density) and `rf_X_Mx`, `rf_X_My`, `rf_X_Mz` (flux components).
RT restart data are written with every output.

### Radiation sources

Sources are currently set in the initial conditions as a spatial source field together with a spectrum.
The spectrum is normalized with one code length, `Rad3D::Length_Unit_CGS(P)`: 1 kpc without `COSMOLOGY` and 1 comoving kpc/{math}`h` with it.
Any new source routine should use the same normalization.

## M1 with Cosmology (`RT_M1` + `COSMOLOGY`)

When `COSMOLOGY` is set, the RT uses Cholla's cosmological code units, the same units as the gas (see {doc}`Cosmology`):

- lengths are comoving kpc/{math}`h`;
- the hydro time step is in super-conformal time, {math}`d\tau = H_0\,dt/a^2`;
- velocities are in units of {math}`v_0 = H_0/h` (km/s), scaled as {math}`v_\mathrm{code} = v_\mathrm{pec}\, a / v_0`.

### Comoving radiation variables

The RT evolves the comoving photon density and flux,

```{math}
\tilde{N} = a^3 N, \qquad \tilde{\mathbf{F}} = a^3 \mathbf{F},
```

the same convention Cholla uses for the gas density.
Photon number is conserved and photon directions do not change in an expanding universe.
In these variables the M1 equations in comoving coordinates {math}`\mathbf{x}` therefore contain no explicit Hubble terms:

```{math}
\frac{\partial \tilde{N}}{\partial t} + \frac{1}{a}\nabla_x \cdot \tilde{\mathbf{F}} = a^3 S - \kappa c \tilde{N},
\qquad
\frac{\partial \tilde{\mathbf{F}}}{\partial t} + \frac{c^2}{a}\nabla_x \cdot \tilde{\mathsf{P}} = -\kappa c \tilde{\mathbf{F}},
```

where {math}`S` is the proper source rate per unit volume, {math}`\kappa` the proper absorption coefficient and {math}`\tilde{\mathsf{P}}` the M1 pressure tensor.
The redshifting of photons from one frequency group into another is neglected.
This is a good approximation when the light-crossing time of the region of interest is short compared with the Hubble time.
The same approximation is made by [Shapiro & Giroux (1987)](https://ui.adsabs.harvard.edu/abs/1987ApJ...321L.107S/abstract).

### Unit conversions

All conversions between RT code units and physical units are made in one place, `Rad3D::Set_Units()`, which is called every step with the current scale factor.
The M1 kernels themselves are the same with and without cosmology.

| Quantity | Without `COSMOLOGY` | With `COSMOLOGY` |
|---|---|---|
| Speed of light in code units | {math}`c / v_\mathrm{unit}` | {math}`c\,a / v_0` |
| Optical depth of a cell | {math}`\sigma\, n_\mathrm{HI}\, \Delta x` | {math}`\sigma\, n_\mathrm{HI,proper}\, a\,\Delta x / h` |
| Source factor applied in the kernel | 1 | {math}`a` |
| Field passed to the chemistry | {math}`N` | {math}`\tilde{N}/a^3` (proper) |

The source factor {math}`a` follows from the time variable: {math}`dt/d\tau \propto a^2` while the code speed of light is proportional to {math}`a`.
The chemistry uses the same cosmological time, density and energy units with and without RT.

### Time stepping and the speed of light

Under `COSMOLOGY` the time step is set by the cosmology and hydro solvers, and the RT never changes it.
A hydro step is typically much longer than the light-crossing time of a cell.
In the Shapiro–Giroux test at {math}`z = 7` one step spans about {math}`10^3` light-crossing times.
The legacy scheme (`rt_reduced_c = 0`) would then give {math}`c_\mathrm{eff} \sim c/10^3`.
Cosmological runs should therefore set `rt_reduced_c > 0`, and the {ref}`3D-Shapiro-Giroux` test uses `rt_reduced_c = 1`.
Check the printed `c_eff/c` and make sure that the ionization fronts of interest move much more slowly than {math}`c_\mathrm{eff}`.

### Snapshots and restarts

RT fields go through the same frame change as the gas, `Grid3D::Change_Cosmological_Frame_System()`.
They are converted forward once after the cosmology is initialized, and back and forth around every write.
On disk the RT fields are stored as comoving quantities, like the gas density.
This convention is set by `RT_DISK_FIELD_ASCALE_POWER` in `src/radiation/radiation.h` (0 = comoving; 3 would write proper fields).

Snapshot times come from the cosmology's {par:param}`scale_outputs_file`; RT output lists are not used.
RT restart files are written and read in the same frame as the snapshots.

### Validation

The {ref}`3D-Shapiro-Giroux` example tests `RT_M1` + `COSMOLOGY` + `CHEMISTRY_GPU` against the analytic cosmological Strömgren sphere of Shapiro & Giroux (1987).
