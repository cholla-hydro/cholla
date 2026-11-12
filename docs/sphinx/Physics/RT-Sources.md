(RT-Sources)=
# Radiation Sources

This page describes the source model that feeds the M1 radiative transfer (see {doc}`RT`).
The model places discrete sources of ionizing photons in the simulation volume, spreads each one over nearby cells with a spatial kernel, and rebuilds the RT source field every hydro step.
Sources can switch on and off, move, and cross the boundaries between MPI subvolumes.

The model is under development.
It has been validated with `RT_M1` + `COSMOLOGY` + `CHEMISTRY_GPU` on the {ref}`3D-Shapiro-Giroux` problem, and supports one source population at a time.

## Requirements

- Build with `RT_M1` and `CHEMISTRY_GPU`, for example with {repository-file}`config/make.type.m1` or {repository-file}`config/make.type.cosmo-m1`.
- Build with `HDF5`: every rank reads its sources from an HDF5 file.
- MPI is optional. With MPI, set `n_proc_x`, `n_proc_y` and `n_proc_z` explicitly, because the source files are split for a specific domain decomposition (see {ref}`rt-sources-splitting`).
- The RT boundaries must be periodic, as for the RT generally.

## How the model works

### Populations and sources

The model separates two things:

- A **population** is a set of sources that share one spectral shape and one spatial kernel.
  It is defined in the parameter file.
  It owns a photo-rate table, built from its spectral shape, and a reference photon rate {math}`L_\mathrm{ref}`.
- A **source** is one member of a population.
  It has a position, a velocity, a luminosity in units of {math}`L_\mathrm{ref}`, a birth time, a lifetime and an integer ID.
  Sources are read from per-rank files when the run starts.
  Each MPI rank stores only the sources inside its own subvolume, on the GPU.

Sources in one population can have different luminosities, positions and lifetimes, but not different spectral shapes.
The reason is how the M1 scheme handles spectra.
All four M1 groups (H I, He I, He II and the unattenuated field) receive the same source field, and they differ only in their opacity.
The chemistry forms effective optical depths from the ratios of the groups, and looks up photoionization and photoheating rates in a table built from one spectrum.
Two different spectra therefore need two tables and two sets of radiation fields.
Only one population (`rt_n_pops = 1`) is supported for now; see {ref}`rt-sources-new-models` for what more would require.

:::{figure} RT-Sources/RT-Sources-structure.svg
:alt: Structure of the RT source model
:width: 100%

How the pieces fit together.
The parameter file defines the population, whose spectrum sets the photo-rate table used by the chemistry.
The catalog is split once, before the run, into one file per rank.
Each step the sources are deposited into the source field `dev_rs`, which drives the M1 radiation fields.
:::

### The source field

Each step the model rebuilds the source field from scratch:

```{math}
r_s(\mathbf{x}_c) = \sum_{i\ \mathrm{alive}} \frac{\ell_i \, W(\mathbf{x}_c - \mathbf{x}_i)}{\Delta V},
\qquad \sum_c W(\mathbf{x}_c - \mathbf{x}_i) = 1,
```

where {math}`\ell_i` is the luminosity of source {math}`i` in units of {math}`L_\mathrm{ref}`, {math}`W` is the kernel weight of cell {math}`c`, and {math}`\Delta V = \Delta x\,\Delta y\,\Delta z` is the cell volume in code units.
The kernel weights of each source sum to exactly 1, so the field integrates to {math}`\sum_i \ell_i` over the volume, and source {math}`i` emits {math}`\ell_i L_\mathrm{ref}` photons per second.

The field is stored in code units (comoving under `COSMOLOGY`) and is not multiplied by the scale factor.
`rtSolve` applies the factor {math}`a` around the RT substeps, as described in {doc}`RT`.
This matches the normalization of the sources the initial conditions used to set, so a source with {math}`\ell = 1` reproduces them exactly.

### What happens each step

The update runs at the start of `Grid3D::Update_RT`, after `Rad3D::Set_Units` and before `rtSolve`:

:::{figure} RT-Sources/RT-Sources-step.svg
:alt: Per-step update of the RT sources
:width: 85%

The per-step update.
Teal steps change the sources; coral steps rebuild the source field.
:::

1. **`Update_RT_State`** copies the grid geometry, the boundary flags and the time of this step into the source model.
2. **Drift** moves each source by its velocity (see {ref}`rt-sources-moving`).
3. **Migrate** hands sources that left the subvolume to the rank that now owns them, one direction at a time (x, then y, then z), so a source that crosses a corner reaches the diagonal neighbor in two hops.
   Positions wrap periodically at the edge of the box.
4. **Deposit** zeroes the source field and adds each live source with its kernel.
   Kernels may extend into ghost cells.
5. **Ghost fold** adds what landed in each ghost cell into the active cell of the rank that owns that location, then zeroes the ghost cells.
6. **`rtSolve`** advances the M1 radiation fields with the new source field.

:::{figure} RT-Sources/RT-Sources-fold.svg
:alt: Ghost-cell fold of a kernel that crosses a subvolume face
:width: 100%

The ghost fold for a CIC source just below a face between two ranks.
Rank A deposits 0.6 into its last active cell and 0.4 into its first ghost cell, which is the same place in space as rank B's first active cell.
The fold moves the 0.4 to rank B.
Folding x, then y, then z also carries corner contributions to the diagonal neighbor.
:::

The fold is exact: the total of the source field is the same before and after, on any number of ranks.
It requires every kernel to reach at most `n_ghost` cells beyond the cell that contains the source (2 for `PCM`, 3 for `PLMP`/`PLMC`, 4 for `PPMP`/`PPMC`), and the code checks this at startup.

## Parameters

Source parameters are read when the source model is initialized.
Only `rt_pop0_*` keys are read while `rt_n_pops` is at most 1.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `rt_n_pops` | int | `0` | Number of source populations. `0` keeps the source field set by the initial conditions (or the restart file); `1` turns on the source model. Any other value is an error. |
| `rt_pop0_sed` | string | `monochromatic_HI` | Spectral shape; see {ref}`rt-sources-seds`. |
| `rt_pop0_sed_T` | float | `1e5` | Blackbody temperature in K. Used only with `rt_pop0_sed = blackbody`. |
| `rt_pop0_Lref` | float | required | Ionizing photon rate {math}`L_\mathrm{ref}` in photons/s of a source with `lum = 1`. Must be positive. |
| `rt_pop0_kernel` | string | `cic` | Spatial kernel: `ngp`, `cic`, `tsc` or `tophat`; see {ref}`rt-sources-kernels`. |
| `rt_pop0_kernel_radius` | float | `1` | Radius of the `tophat` kernel in cells. Must be at least 0.5 and at most `n_ghost`. Ignored by the other kernels. |
| `rt_pop0_rule` | string | `per_rank_file` | How sources are created. `per_rank_file` is the only rule so far. |
| `rt_pop0_catalog` | string | required | File prefix: rank {math}`p` reads `<prefix>.h5.<p>`. Required even on a restart, where the sources are read from the restart file instead. |

A typical block for a single blackbody population:

```ini
# RT sources
rt_n_pops=1
rt_pop0_sed=blackbody
rt_pop0_sed_T=1e5
rt_pop0_Lref=5e48
rt_pop0_kernel=cic
rt_pop0_rule=per_rank_file
rt_pop0_catalog=sources/my_sources
```

With `rt_n_pops = 1`, any source field and spectrum set by the initial conditions are ignored, and the population's spectrum replaces the photo-rate table.
The initial conditions still set up the gas.
This lets the existing RT test problems run with sources from the model.

At startup the code prints one line describing the population and the total number of sources, for example:

```text
RT sources: 1 population(s); pop 0: sed=monochromatic_HI Lref=5.000000e+48 photons/s kernel=cic, 1 sources.
```

(rt-sources-seds)=
### Spectral shapes

| `rt_pop0_sed` | Spectrum |
|---|---|
| `monochromatic_HI` | All photons at the H I ionization threshold. This is the spectrum of the Iliev Test 1 and the Shapiro–Giroux problem. |
| `blackbody` | A blackbody at `rt_pop0_sed_T`, as in Iliev Test 5 ({math}`10^5` K). |
| `popII` | Population II stellar spectrum used in CROC. |
| `popIII` | Population III stellar spectrum used in CROC. |
| `quasar` | Average quasar spectrum used in CROC. |

The shapes come from `SpectralShape` in {repository-file}`src/radiation/alt/spectral_shape.h`, which also gives the references for the `popII`, `popIII` and `quasar` shapes.
Each shape is normalized to a unit ionizing photon rate and scaled by {math}`L_\mathrm{ref}`, with lengths in `Rad3D::Length_Unit_CGS` (1 kpc without `COSMOLOGY`, 1 comoving kpc/{math}`h` with it).
For `monochromatic_HI` the code applies the same 6.34/5.92 correction as the initial conditions of the Iliev tests, because the threshold frequency bin is centered slightly above the threshold.

(rt-sources-kernels)=
### Spatial kernels

Let {math}`u` be the source position along one axis in cell units, so that cell {math}`i` spans {math}`[i, i+1)` and has its center at {math}`i + \tfrac12`.
The `ngp`, `cic` and `tsc` kernels are products of the one-dimensional weights below.

| `rt_pop0_kernel` | Cells per axis | Weights along one axis | Reach beyond the source's cell |
|---|---|---|---|
| `ngp` | 1 | 1 in the cell that contains the source | 0 |
| `cic` | 2 | {math}`1 - f` and {math}`f` in the two cells whose centers bracket the source, where {math}`f` is the fractional distance from the lower center | 1 |
| `tsc` | 3 | {math}`\tfrac12(\tfrac12 - d)^2`, {math}`\tfrac34 - d^2`, {math}`\tfrac12(\tfrac12 + d)^2`, where {math}`d \in [-\tfrac12, \tfrac12)` is the offset from the center of the source's cell | 1 |
| `tophat` | varies | Equal weights in every cell whose center lies within `rt_pop0_kernel_radius` cells of the source (all cells in a sphere). If no center is that close, the source falls back to `ngp`. | {math}`\lceil R \rceil` |

A `cic` source exactly on a cell corner gives {math}`1/8` to each of the eight cells around the corner.
That is the source the Iliev and Shapiro–Giroux initial conditions place at the box center.

Larger kernels avoid a single very bright cell and make the source less sensitive to its position within a cell.
They also spread the photons over a larger volume, which matters if the Strömgren radius is only a few cells.

## Source catalogs

### The CSV catalog

A catalog is a CSV file whose first row names the columns.
Columns may be in any order, names are not case sensitive, and lines that start with `#` are ignored.

| Column | Required | Default | Meaning |
|---|---|---|---|
| `x`, `y`, `z` | yes | | Position in code length units: kpc without `COSMOLOGY`, comoving kpc/{math}`h` with it. Positions outside the box are wrapped periodically. |
| `lum` | yes | | Luminosity in units of `rt_pop0_Lref`. Must not be negative. |
| `vx`, `vy`, `vz` | no | `0` | Velocity; see {ref}`rt-sources-moving` for the units. |
| `t_birth` | no | `-1e30` | Time at which the source switches on, in Myr. The default means the source is always on. |
| `t_life` | no | `0` | Lifetime in Myr. A value {math}`\le 0` means the source never switches off. |
| `id` | no | row index | Integer ID, unique within the catalog. Rows are numbered from 0 if the column is missing. |

For example, the single source of the Shapiro–Giroux problem, at the center of a 2000 kpc/{math}`h` box:

```text
x,y,z,vx,vy,vz,lum,t_birth,t_life,id
1000.0,1000.0,1000.0,0.0,0.0,0.0,1.0,-1.0,0.0,0
```

(rt-sources-splitting)=
### Splitting a catalog with `rt_split_sources.py`

Cholla never reads the global catalog.
Each rank reads only its own file, so the cost of reading sources does not grow with the number of ranks.
The script {repository-file}`python/scripts/rt_split_sources.py` writes those files.
It needs `numpy` and `h5py`.

```shell-session
$ python python/scripts/rt_split_sources.py sg_catalog.csv --param shapiro_giroux_cosmo.txt --prefix sg_source
Split 1 sources from sg_catalog.csv over 8 ranks (2 x 2 x 2) into sg_source.h5.*
  sources per rank: min 0, max 1
```

The script takes the grid and the MPI decomposition from a Cholla parameter file, from the command line, or both; the command line wins.

| Option | Meaning |
|---|---|
| `catalog` | The CSV catalog. |
| `--param FILE` | Cholla parameter file; `nx`, `ny`, `nz`, `xmin`, `ymin`, `zmin`, `xlen`, `ylen`, `zlen`, `n_proc_x`, `n_proc_y` and `n_proc_z` are read from it. |
| `--n NX NY NZ` | Global number of cells. |
| `--min XMIN YMIN ZMIN` | Lower corner of the box. |
| `--len XLEN YLEN ZLEN` | Size of the box. |
| `--nproc PX PY PZ` | Number of ranks along each axis. |
| `--prefix PREFIX` | Output prefix. Rank {math}`r` gets `PREFIX.h5.r`; directories in the prefix are created. Use the same prefix for `rt_pop0_catalog`. |

The script reproduces Cholla's domain decomposition, including the uneven split when the number of cells is not divisible by the number of ranks, and Cholla's numbering of ranks (x fastest, then y, then z).
It does not reproduce Cholla's automatic choice of the decomposition, which is why `n_proc_x`, `n_proc_y` and `n_proc_z` must be given.
It writes a file for every rank, including ranks with no sources.

The script stops with a message if a column is unknown, duplicated or missing, if a row has the wrong number of values or a value is not a number, if a luminosity is negative, or if the IDs are not unique integers.

### The per-rank file format

Each file holds one group, `/sources`.
The RT restart files use the same group, so one reader handles both.

| Name | Kind | Contents |
|---|---|---|
| `n_local` | attribute, int | Number of sources in this file. |
| `n_fields` | attribute, int | Number of values per source (10). |
| `local_min` | attribute, double[3] | Lower corner of the subvolume the file was written for. |
| `local_max` | attribute, double[3] | Upper corner. |
| `data` | dataset, double, `n_local` × 10 | One row per source: `x y z vx vy vz lum t_birth t_life id`. Absent when `n_local` is 0. |

When it reads a file, Cholla checks that `local_min` and `local_max` match the subvolume of the rank, to {math}`10^{-6}` of a cell.
A file split for a different decomposition is refused with a message that names the rank and both subvolumes.
Cholla also checks every source position, allowing one cell of slack at the faces for rounding; a source in that slack region is handed to its owner by the first migration, before anything is deposited.

### Moving a catalog to a different decomposition

Re-run `rt_split_sources.py` with the new `n_proc_x`, `n_proc_y` and `n_proc_z`.
The sources in a restart file are tied to the decomposition they were written with, so a restart must use the same decomposition (see {ref}`rt-sources-restarts`).

## Making sources appear and disappear

A source emits only while

```{math}
t_\mathrm{birth} \le t_\mathrm{mid} < t_\mathrm{birth} + t_\mathrm{life},
```

where {math}`t_\mathrm{mid}` is the time at the middle of the current hydro step, in Myr.
If {math}`t_\mathrm{life} \le 0` there is no upper limit.

:::{figure} RT-Sources/RT-Sources-lifetime.svg
:alt: Which steps a source emits in
:width: 100%

A source emits at its full luminosity in every step whose midpoint lies in its window, and not at all in other steps.
:::

The meaning of {math}`t` depends on the build:

- Without `COSMOLOGY`, {math}`t` is the simulation time converted to Myr (the code time unit is the kyr).
- With `COSMOLOGY`, {math}`t` is the cosmic time since the Big Bang that Cholla integrates from the scale factor.
  Cholla prints it at startup as `Revised t_start : ... Myr` and every step as `t_physical`, which helps in choosing birth times.

To have a source appear during a run, give it a `t_birth` later than the start time.
To have it disappear, give it a positive `t_life`.
All sources must be in the catalog when the run starts; there is no way yet to add sources during a run except through their birth times.

Because emission is decided per step, a source's photon budget can be off by up to one hydro step at each end of its life.
If that matters, keep the time step short compared with the lifetimes, for example with `max_timestep` without `COSMOLOGY` or a smaller `max_delta_a` with it.

Sources that have not been born yet, and sources that have died, stay in memory.
They still drift and migrate, but cost nothing in the deposit.

(rt-sources-moving)=
## Moving sources

Every source moves with a constant velocity.
Each step its position changes by {math}`\mathbf{v}\,\times` `drift_factor`:

- Without `COSMOLOGY`, `drift_factor` is the hydro time step, so velocities are in code units, kpc/kyr ({math}`\approx 978` km/s).
- With `COSMOLOGY`, `drift_factor` is the position drift that Cholla's dark-matter particles use in the first half of their kick–drift–kick step (`Advance_Particles_KDK_Step1_Cosmo_Kernel`), so velocities use the same units and conventions as the particle velocities.

Source velocities are never kicked, so a source keeps its velocity in code units.
In a cosmological run that differs from a freely streaming particle, whose peculiar velocity decays as the universe expands, and moving sources have not yet been validated with `COSMOLOGY`.
For sources that should follow the matter, the better long-term route is a source rule that attaches sources to particles (see {ref}`rt-sources-new-models`).

After drifting, sources that left the subvolume are sent to the rank that owns their new position, and positions wrap periodically at the edges of the box.
Migration moves a source by at most one subvolume per direction per step, which is far more than any realistic source travels.
With non-periodic boundaries a source that leaves the box is dropped, and the code reports how many were dropped; the RT itself currently requires periodic boundaries.

(rt-sources-restarts)=
## Restarts and outputs

On a restart the sources are read from the `/sources` group of each rank's RT restart file, not from the catalog, so their positions, lifetimes and IDs continue where they left off.
The restart must use the same decomposition as the run that wrote it.
`rt_pop0_catalog` must still be set, and the spectrum is rebuilt from the parameters.
The source field itself is rebuilt on the first step.

Snapshots written with `OUTPUT_RADIATION` contain the source field as `rf_sources`, in the code units described above.
Summing `rf_sources` {math}`\times\,\Delta V` over the active cells gives {math}`\sum_i \ell_i` for the sources that were alive in the last step; multiply by {math}`L_\mathrm{ref}` for the photon rate.

## Testing

Unit tests for the deposit, the ghost fold, the lifetimes and the migration are in {repository-file}`src/radiation/sources/rt_sources_tests.cu`.
Build them with `make TEST=true TYPE=cosmo-m1` (see {doc}`../Development/testing`) and run the `tRTSources.*` tests.
They check that:

- a `cic` source on a cell corner reproduces the 8-cell source of the initial conditions;
- every kernel conserves the luminosity to round-off when its footprint wraps across a periodic corner, and leaves the ghost cells empty;
- a source emits only within its lifetime window;
- a source that drifts out through one face comes back in through the opposite face.

For an end-to-end check, run the {ref}`3D-Shapiro-Giroux` problem twice: once with `rt_n_pops = 0`, and once with `rt_n_pops = 1`, `rt_pop0_sed = monochromatic_HI`, `rt_pop0_Lref = 5e48`, `rt_pop0_kernel = cic` and the one-source catalog above.
The two runs should agree to round-off.
On 2×2×2 ranks the source sits on the corner shared by all eight subvolumes, so the second run also exercises the ghost fold across MPI.

(rt-sources-new-models)=
## Adding new source models

### Where the code lives

| File | Contents |
|---|---|
| {repository-file}`src/radiation/rt_source_state.h` | `RT_State`: grid geometry, boundary flags and time for the current step. |
| {repository-file}`src/radiation/sources/rt_sources.h`, `rt_sources.cpp` | `RTSourceModel`: parameters, initialization and the per-step `Update`. |
| {repository-file}`src/radiation/sources/rt_source_population.h` | `RTPopulation` and the enums for spectra, kernels and rules. |
| {repository-file}`src/radiation/sources/rt_source_sed.cpp` | `Set_Population_SED`: builds the spectral shape and updates the photo-rate table. |
| {repository-file}`src/radiation/sources/rt_source_store.h`, `rt_source_store.cu` | `SourceStore`: sources on the GPU, drift and MPI migration. |
| {repository-file}`src/radiation/sources/rt_source_apply.cu` | Deposit kernels and the ghost fold. |
| {repository-file}`src/radiation/sources/rt_source_io.cpp` | Per-rank HDF5 reader and restart writer. |
| {repository-file}`python/scripts/rt_split_sources.py` | Splits a CSV catalog into per-rank files. |

The source model is called from `Grid3D::Update_RT` and `Grid3D::Update_RT_State` in {repository-file}`src/radiation/radiation.cpp`, and initialized in {repository-file}`src/main.cpp`.

### General guidelines

- **Keep the work local.**
  Each rank should only touch the sources it owns, and nothing should read or build a global list of sources during a run.
  This is what lets the model scale to large runs.
- **Use enums and `switch` on the GPU.**
  The device code selects kernels and spectra with integer enums and `switch` statements, not virtual functions, which work poorly across the host–device boundary in CUDA and HIP.
  Keep the integer values of existing enum members stable.
- **Keep the source field in code units.**
  Write {math}`\ell\,W/\Delta V` without any factor of the scale factor; `rtSolve` applies it.
- **Read every parameter in `RTSourceModel::Read_Parameters_`.**
  Cholla only accepts parameters that some code reads, and warns about the rest.
  Build per-population keys from the population index, as the existing `rt_pop0_*` keys are.
- **Add a unit test** to `rt_sources_tests.cu` for every new kernel or rule, and check conservation.

### A new spectral shape

1. Add a member to `SedType` and to `Parse_Sed_Type` and `Sed_Type_Name` in `rt_sources.cpp`.
2. Add a `case` to `Set_Population_SED` that fills the shape on the cross-section frequency grid, normalized to a unit ionizing photon rate (as `SpectralShape::Normalize` does).
   The common code then scales it by {math}`L_\mathrm{ref}` and updates the table.
3. Add any new parameters (such as `rt_pop0_sed_T` for the blackbody) to `RTPopulation` and `Read_Parameters_`.

A spectrum that changes with time, for example an aging stellar population, must change for the whole population at once, because the population shares one table.
Call `Set_Population_SED` from `RTSourceModel::Update` when the shape has changed enough to matter.
The table is rebuilt on the GPU, but it is not free, so avoid rebuilding it every step without need.
Sources whose spectra age independently belong in separate populations.

### A new spatial kernel

1. Add a member to `KernelType` with a new integer value, and add it to `Parse_Kernel_Type`, `Kernel_Type_Name` and `Kernel_Half_Width`.
2. Add the weights to `Deposit_Kernel` in `rt_source_apply.cu`: in `Axis_Weights` for a kernel that is a product of one-dimensional weights, or as a separate branch, like the `tophat`, otherwise.
3. Make sure the weights of one source sum to exactly 1 over its whole footprint, wherever the source sits in its cell.
   Normalize over the full footprint, never over the part that falls on one rank.
4. Make sure the reach returned by `Kernel_Half_Width` is correct; the startup check against `n_ghost` relies on it.
   A kernel that must reach further than `n_ghost` needs a different ghost exchange, such as depositing on behalf of neighboring ranks.
5. Add the kernel to the conservation test.

### A new rule for creating sources

New rules are how sources should be created from the simulation itself, for example at star particles, at dark-matter halos, or from a local abundance model.

1. Add a member to `RuleType`, to `Parse_Rule_Type`, and to the parameter reading.
2. Create the sources on each rank from local data only, and add them with `SourceStore::Append_Device` (or `Append_Host`).
   New sources may be created anywhere in the subvolume or its one-cell margin.
3. Do the creation in `RTSourceModel::Update` before `Migrate`, so that any source created outside the subvolume reaches its owner before the deposit.
4. Give each source a unique ID without communication, for example from a particle ID or from a per-rank counter combined with the rank number.
5. Sources that should follow particles can copy the particle positions and velocities each step instead of drifting on their own.
6. If sources are created and destroyed often, remove dead sources from the store from time to time.
   The partition kernel in `rt_source_store.cu`, which already separates the sources that stay from those that leave, is a good starting point.
7. Write new sources to the restart file through the existing `/sources` group, so a restart continues exactly.

### More than one population

The layout of the source field already allows one field per population (`dev_rs` has `max(1, rt_n_pops)` fields), and each population carries its own index.
Supporting `rt_n_pops > 1` also requires:

- a separate set of radiation fields for each population, because the group ratios the chemistry uses are only meaningful for one spectrum (16 more fields per population with `RT_M1`);
- one photo-rate table per population (the table wrapper is currently created with a single table in the `Rad3D` constructor);
- summing the photoionization and photoheating rates over the populations in the chemistry;
- either one `SourceStore` per population, or depositing each population with only its own sources;
- relaxing the checks on `rt_n_pops` in {repository-file}`src/global/global.cpp` and in `RTSourceModel::Initialize`.

Each population adds 17 grid fields, so check the memory budget before adding many.
