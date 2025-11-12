# Output Format & Repacking

## Basic Background

Cholla simulations run on a spatial domain corresponding to a rectangular prism.
Cholla primarily deals with 2 types of data:

- **grid-data**: Regularly spaced data composed of a large grid/mesh of equal-sized cells.
  During simulations, we commonly call this "field" data.
  The canonical example of this datatype is the collection of finite-volume fields used for hydrodynamics.
  These are tracked at cell-centers.
  There are also other kinds of fields (e.g. cell-face-centered magnetic fields used in Constrained Transport).

- **particle-data:** Cholla can also track a collection of "particles," where each particle has unique set of properties.
  These properties always include an id, position, and velocity.
  They can also have other properties (e.g. mass).
  These are commonly used to model dark matter or star-clusters.

## Domain Partitioning

The simulation domain is partitioned into 1 or more equal volume (non-overlapping) "blocks."
The blocks are conceptually organized on a 3D "block-location" grid, whose shape we denote as `(BLx, BLy, BLz)`.
When a simulation starts, this shape is always chosen to match `(n_proc_x, n_proc_y, n_proc_z)`.
Each process in a simulation evolves the contents for a separate block.

How decomposition relates to the data-types:

- **grid-data:** conceptually all fields live on a global Domain-grid composed of `(nDx,nDy,nDz)` cells.
  In practice, the underlying field data is partitioned among the blocks.
  A given block is responsible for tracking data on `(nBx,nBy,nBz)` cells, where `(nDx,nDy,nDz) = (BLx*nBx, BLy*nBy, BLz*nBz)`.
- **particle-data:** each block tracks a separate list of particle properties

## File Format Overview

Cholla can write many kinds of datasets.
The standard kinds of datasets broadly fall into 2 categories:

- snapshot-data, which is a "snapshot" of the simulation state (and is required for restarts).
  This consists of general field-data, gravity-data, and particle-data.

- derived data on 2D grids (e.g. slices/projections).

Each time Cholla writes an output, it writes a separate file for each data-kind and for each block.
For example, a simulation with 16 processes that writes 5 data-kinds at a given cycle will produce `5*16=80` files.
We provide [scripts](#concatenation-scripts) to concatenate (or consolidate) this data into fewer files.

By default, Cholla writes files in the hdf5 format.
HDF5 files store data as "attributes" (header-variables describing the data) and "datasets" which stores the data itself.
HDF5 provides a lot of flexibility for organizing this information in a hierarchy of "groups" (similar to how you can organize different files in a hierarchy of directories).


:::{note}
For certain kinds of snapshot-data Cholla stores data in memory with "ghost zones".
"Ghost zones" are **NEVER** written to disk.
:::

:::{note}
This primarily pertains to 3D data:

- while the z-axis is generally the fastest access index in the datafile, the x-axis is generally the fastest access index in the simulations

- the gravity array has historically been stored as a 1D array.
  I suspect that the data was not reordered (so the memory layout is "native").
:::

### Flat (Default) Schema:

Cholla's standard "schema" (strategy for organizing data) is "flat" (i.e. there is no hierarchy).
We represent this schema below:
```
/                                      # root group
 ├── HEADER-ATTRS  (REQUIRED)
 ├── <dataset-0>
 ├── <dataset-1>
 └── ...
```

At the time of writing, all files written by Cholla use this schema.
Historically all concatenation scripts would use this schema.

### Hierarchical Schema

We adopt variants an alternative hierarichal schema when we repack snapshot data.
There are currently 2 variants of this schema (for grid-data and particle-data).

We preface our description by mentioning that **ALL** variants **MUST** have an HDF5 group called `"domain"`, which is described by the structure:

```
domain/
 ├── blockid_location_arr     # shape: (BLx,BLy,BLz)
 └── stored_blockid_list      # shape: (nBStored,)
```
In the above diagram (and in subsequent diagrams), 
- we follow numpy conventions for describing arrays with C-contiguous layouts.
  In other words, the fastest index is last axis

- we define `BLx`,`BLy`, and `BLz` as the number of blocks per axis.

- `nBStored` is the number of blocks in the file.
  It should nominally be `1` or `BLx*BLy*BLz`

- `"domain/blockid_location_arr"` specifies the relative locations of the blocks.


With that out of the way, we now describe both variants:

:::{tab} grid-data

Presently, this is the scheme produced by scripts that concatenate 3D field-data or {ref}`repacking previously concatenated field data. <repack-script>`

```
/                                  # root group
 ├── HEADER-ATTRS  (REQUIRED)
 ├── domain/       (REQUIRED)
 │    ├── blockid_location_arr     # shape: (BLx,BLy,BLz)
 │    └── stored_blockid_list      # shape: (nBStored,)
 └── field/
      ├── <field-0>     # 4D shape: (nBStored,nBx,nBy,nBz) or (nBStored, ...)
      ├── <field-1>
      └── ...
```

In the above diagram:

- `nBx`,`nBy`,`nBz` refer to the number of cells per block.
  This is the shape of a cell-centered field.
  Don't confuse these with `BLx`, `BLy`, and `BLz`, (which we previously defined as the number of blocks per axis)

- the data at `{field/<field-0>}[i, ...]` corresponds to the data of the block with blockid specified by `{domain/stored_blockid_list}[i]`

```{note}
Files in this format **ALWAYS** provide the `"dims"` attribute (in HEADER-ATTRS) and the ``"domain"`` group.
Importantly, `"dims"` specifies `(nDx,nDy,nDz)`, the number of cells on the conceptual global Domain-grid, and the shape of `"domain/blockid_location_arr"` is `(BLx,BLy,BLz)`.
Thus, you always can infer `(nBx,nBy,nBz) = (nDx/BLx, nDy/BLy, nDz/BLz)`.
Consequently, you can determine whether a `field/<field-0>` is cell-centered, face-centered, etc. by looking at the field's shape.
```
:::


:::{tab} particle-data

Presently, this is the scheme produced by scripts that concatenate particle data

```
/                                      # root group
 ├── HEADER-ATTRS  (REQUIRED)
 ├── domain/       (REQUIRED)
 │    ├── blockid_location_arr        # shape: (BLx,BLy,BLz)
 │    └── stored_blockid_list         # shape: (nBStored,)
 └── particle/
      ├── <ptype-a>
      │    ├── ATTR:total_ptype_count    # i64
      │    ├── stop_block_idx_slc        # 1D shape: (nBStored,)
      │    ├── <particle-prop-a0>        # 1D shape: (stop_block_idx_slc[-1],)
      │    ├── <particle-prop-a1>        # 1D shape: (stop_block_idx_slc[-1],)
      │    └── ...                       # 1D shape: (stop_block_idx_slc[-1],)
      ├── <ptype-b>
      │    ├── ATTR:total_ptype_count    # i64
      │    ├── stop_block_idx_slc        # 1D shape: (nBStored,)
      │    ├── <particle-prop-b0>        # 1D shape: (stop_block_idx_slc[-1],)
      │    ├── <particle-prop-b1>        # 1D shape: (stop_block_idx_slc[-1],)
      │    └── ...                       # 1D shape: (stop_block_idx_slc[-1],)
      └── ...
```
In the above diagram:

- Although at the time of writing, Cholla only ever considers a single particle-type, the schema is structured to accomodate the use of multiple particle-types in a single simulation (e.g. dark-matter particles, star cluster particles, *maybe* tracer particles)

- **IMPORTANTLY:** different particle types are allowed to have different sets of attributes.
  For the sake of example, we may want to explicitly track creation time & mass for cluster particles (since they can vary from particle-to-particle), but it makes no sense to do the same for dark matter particles.

- `particle/<ptype-a>/ATTR:total_ptype_count` specifies the total number of particles (of the given type) in the ENTIRE simulation.

- `particle/<ptype-a>/stop_block_idx_slc` holds monotonically non-decreasing values.
  It also satisfies the invariants that
  - `{particle/<ptype-a>/stop_block_idx_slc}[0] >= 0`
  - When `nBStored == nBx*nBy*nBz`, then `{particle/<ptype-a>/stop_block_idx_slc}[-1] =={particle/ATTR:total_ptype_count}`.
    In other cases, `{particle/<ptype-a>/stop_block_idx_slc}[-1] <= {particle/<ptype-a>/ATTR:total_ptype_count}`

- The values of ``<ptype-a>/<particle-prop-a0>`` that describe particles for the blockid specified by `{domain/stored_blockid_list}[i]` are given by `{particle/<ptype-a>/<particle-prop-a0>}[slc]`, where `slc` is:
  - ``0:{particle/<ptype-a>/stop_block_idx_slc}[0]``, when `i` is 0
  - ``{particle/<ptype-a>/stop_block_idx_slc}[i-1]:{particle/<ptype-a>/stop_block_idx_slc}[i]``, in all other cases


```{note}
If we're interested in supporting cases where a simulation was run with multiple kinds of particle-data, we may want to store an attribute in the ``particle`` HDF5 group to explicitly list all known particle-types.
```

:::

#### ASIDE: Preview of a hypothetical, more general schema

Both of the above variants are intended to be forward-compatible with a **hypothetical** variant that can store field-data, particle-data, and gravity-data in the same file.
We illustrate what this could look like down below

```
/                                      # root group
 ├── HEADER-ATTRS  (REQUIRED)
 ├── domain/       (REQUIRED)
 │    ├── blockid_location_arr        # shape: (BLx,BLy,BLz)
 │    └── stored_blockid_list         # shape: (nBStored,)
 ├── field/
 │    ├── <field-0>        # 4D shape: (nBStored,nBx,nBy,nBz) or (nBStored, ...)
 │    ├── <field-1>        # 4D shape: (nBStored,nBx,nBy,nBz) or (nBStored, ...)
 │    └── ...              # 4D shape: (nBStored,nBx,nBy,nBz) or (nBStored, ...)
 ├── particle/
 │    ├── <ptype-a>
 │    │    ├── ATTR:total_ptype_count    # i64
 │    │    ├── stop_block_idx_slc        # 1D shape: (nBStored,)
 │    │    ├── <particle-prop-a0>        # 1D shape: (stop_block_idx_slc[-1],)
 │    │    ├── <particle-prop-a1>        # 1D shape: (stop_block_idx_slc[-1],)
 │    │    └── ...                       # 1D shape: (stop_block_idx_slc[-1],)
 │    ├── <ptype-b>
 │    │    ├── ATTR:total_ptype_count    # i64
 │    │    ├── stop_block_idx_slc        # 1D shape: (nBStored,)
 │    │    ├── <particle-prop-b0>        # 1D shape: (stop_block_idx_slc[-1],)
 │    │    ├── <particle-prop-b1>        # 1D shape: (stop_block_idx_slc[-1],)
 │    │    └── ...                       # 1D shape: (stop_block_idx_slc[-1],)
 │    └── ...
 └── gravity/
      └── gravity          # 4D shape: (nBStored,nBx,nBy,nBz)
```

## Header Attributes
The following attributes are attached to all Cholla outputs:
- 'gamma', the ratio of specific heats that the simulation was run with
- 't', the time of the snapshot, in code units (usually kyr)
- 'dims', a 3-element attribute that gives the number of cells per axis in the x, y, and z directions for the conceptual grid that spans the entire domain
- 'n\_step', the simulation step when the data was output

Additional attributes that are attached to newer Cholla outputs include:
- 'dx', a three dimensional attribute that gives the x, y, and z dimensions of a cell, in code units,

and a series of "unit" attributes, that provide the conversion between whatever units the code was run in and cgs:
- 'length\_unit'
- 'time\_unit'
- 'mass\_unit'
- 'density\_unit'
- 'velocity\_unit'
- 'energy\_unit'

So, for example, if the code was run with a mass unit of one solar mass and a length unit of one kpc, and you have read the density into an array called 'd', multiplying d by the density unit would convert the density array to g/cm{sup}`3`.

## Field Data
:::{tip}
In files using the data hierarchical-format, this data is all stored within the "field" hdf5 group.
:::

Datasets contain the different fields that are evolved by the simulation.
They are 1, 2, or 3 dimensional arrays, corresponding to the dimensionality of the simulation (and specified by the nx, ny, nz attributes, as described above).
The following are the conserved variable fields that are always outputs for a hydrodynamic simulation (all output in code units):
- 'density', the mass density in each cell (i.e. M{sub}`⊙` / kpc{sup}`3`)
- 'momentum\_x', the x-momentum density
- 'momentum\_y', the y-momentum density
- 'momentum\_z', the z-momentum density
- 'Energy', the total energy density

If Cholla is run with the dual energy flag ('DE'), the thermal energy field will also be present:
- 'GasEnergy', the thermal energy density, equivalent to the total energy density minus the kinetic energy density.

If Cholla is run with the passive scalar flag ('SCALAR'), a number of scalar fields may also be present, e.g.:
- 'scalar0', the value of the first passive scalar.

If Cholla is run with 'DUST':
- 'dust\_density', the dust density in code units (M{sub}`⊙` / kpc{sup}`3`)

If an MHD simulation is run, Cholla writes the following face-centered fields:
- 'magnetic\_x'
- 'magnetic\_y'
- 'magnetic\_z'

## Slices, Projections, and Rotated Projections
For 3D simulations, Cholla can also be run with flags to output slices and projections of the data.
(This can be useful for larges simulations if saving the full dataset is too costly to achieve a high time resolution for snapshots.)
The relevant make flags are 'SLICES', 'PROJECTIONS', and 'ROTATED\_PROJECTIONS'.
All produce hdf5 files similar to full grid outputs, with the same header attributes.
Datasets present in slices include all the conserved variables, as well as the thermal energy density and scalars, if the simulation was run with them.
Three slices will be output: 'xy' slices (a slice along the z-midplane), 'xz' slices (and a slice along the y midplane), and 'yz' slices (a slice along the x midplane).
Datasets in the hdf5 file are named according to which direction the slice was made.
For example, datasets in a 'slice\_xy' file will include:
- 'd\_xy', the mass density in cells along the z-midplane of the simulation
- 'mx\_xy'
- 'my\_xy'
- 'mz\_xy'
- 'E\_xy'
- 'GE\_xy' (if DE is on)
- 'scalar\_xy' (if SCALAR is on)

Projections are similar to slices in that they are 2 dimensional datasets, but are integrated along the relevant direction.
Currently, Cholla outputs density projections, and density-weighted temperature projections (density times temperature for a given cell).
Both are output in code units.
So, for example, if the code was run with density units of M{sub}`⊙`/kpc{sup}`3`, a density projection output will have units of M{sub}`⊙` / kpc{sup}`2` and a temperature projection will have units of M{sub}`⊙` K / kpc{sup}`2`.
The PROJECTIONS flag outputs xy and xz projections (integrated along the z and y axes, respectively).
Datasets are called:
- 'd\_xy'
- 'T\_xy'
- 'd\_xz'
- 'T\_xz'
- 'd\_dust\_xy', (if 'DUST' was used)
- 'd\_dust\_xz', (if 'DUST' was used)

Rotated projections are similar, but are integrated along an axis specified by the input parameter file (see the relevant wiki page for details).

## Particle data

In all simulations with particles, HDF5 files will include a collection of 1D datasets that are each named after the particle-property that is tracked.
The collection of datasets or arrays is equivalent to the ["parallel array" organization strategy](https://en.wikipedia.org/wiki/Parallel_array).
In other words, values located at the same index in each dataset all describe different attributes of the same particle.

While the precise set of particle properties vary between simulations, particles always have the following attributes:

- ``particle_IDs`` (the particle id that is unique to the current particle)
- ``pos_x``, ``pos_y``, ``pos_z`` (tracks positions)
- ``vel_x``, ``vel_y``, ``vel_z`` (tracks particle velocity)

:::{todo}
Expand on other attributes.
:::

At the time of writing, Cholla assumes there is only one particle-type in a simulation and directly writes the property datasets using the flat scheme.
Under the alternative hierarchical-format, the property datasets are found under the ``particle/<ptype>/`` HDF5 grouping.


## Gravity data

At the time of writing, this is just used for restarts

## Scripts

We provide a variety of scripts for modifying outputs in the directory called {repository-file}`python/scripts`.

:::{todo}
We may want to directly embed the result of each script's help command within the documentation.
:::

### Concatenation scripts

The following scripts are provided (for use as command-line tools or as python modules) to help with concatenation:

- {repository-file}`python/scripts/concat_2d_data.py`, for concatenating 2D datasets such as slices, projections, and rotated projections
- {repository-file}`python/scripts/concat_3d_data.py`, for concatenating field data (aka 3D datasets)
- {repository-file}`python/scripts/concat_particles.py`, for concatenating particle datasets

#### What is concatenation?

As [noted above](#file-format-overview), whenever Cholla outputs data, a separate file is written for each data-kind and for each block.
Concatenation is the process **(for a single data-kind)** of consolidating the contents of the files for all of the blocks into a single file.
Historically, this procedure would produce an HDF5 file with the "Flat Format," where the data is stitched together in a way that roughly approximates the file that would be produced by a simulation run with a single block that spanned the entire domain.
This is indeed still the behavior of `concat_2d_data.py` and `concat_particles.py`.

However, more recent versions of cholla include versions of `concat_3d_data.py` that will produce files HDF5 files in the "Hierarchical Format."


#### CLI Usage

The CLI for all the scripts is similar and details can be found when passing the `--help` option to the script.
In general you need to tell the script which directory to read files from (the `-s`/`--source-directory` flag), where to write the concatenated files (the `-o`/`--output-directory` flag), and which outputs to concatenate (the `--snaps` flag).
The `--snaps` flag accepts a couple of different input formats, it can be a single number (e.g. 8), a range (e.g. 2-9), or a list (e.g. [1,2,3]); ranges are inclusive.

We show the detailed help messages down below

::::{tab} concat\_2d\_data.py

:::{include-cli-help} ../../python/scripts/concat_2d_data.py
:::

::::

::::{tab} concat\_3d\_data.py

:::{include-cli-help} ../../python/scripts/concat_3d_data.py
:::

::::


::::{tab} concat\_particles.py

:::{include-cli-help} ../../python/scripts/concat_particles.py
:::

::::

**Example**
```bash
./concat_3d_data.py -s /PATH/TO/SOURCE/DIRECTORY/ -o /PATH/TO/DESTINATION/DIRECTORY/ -n 8  --snaps 0-10
```

#### Import Usage

The scripts above contain three public functions, `concat_2d_dataset`, `concat_3d_dataset`, and `concat_particles_dataset`.
These functions will each concatenate a single output time of a 2D, 3D or particle dataset respectively and can be imported into another python program assuming the scripts are in your python path.
Generally the easiest way to import this script is to add the `python/scripts` directory to your python path in your script like this:

```python
import sys
sys.path.append('/PATH/TO/CHOLLA/python/scripts')
import concat_3d_data
```

(repack-script)=
### Repack

Next, we turn our attention to the script called {repository-file}`python/scripts/snaprepack.py`.
This file is intended to be used to repack a previously concatenated snapshot file.
The output file will use the Hierarchical Format.

Ideally, you can use this script by invoking:
```bash
./snaprepack.py -s PATH/TO/SOURCE/FILE.h5 -o /PATH/TO/OUTPUT/DIRECTORY
```

The above will produce an error if the file is missing the ``"nprocs"`` attribute from the header.
In that case, you can supply it via the ``--missing-nprocs-triple`` argument.
For example, if ``"nprocs"`` should hold ``(4,2,2)``, then you could pass

```bash
./snaprepack.py -s PATH/TO/SOURCE/FILE.h5 -o /PATH/TO/OUTPUT/DIRECTORY --missing-nprocs-triple 4 2 2
```
**NOTE:** we currently do **NOT** support overriding the value of "nprocs".

For more details about options, invoke ``snaprepack.py --help``
