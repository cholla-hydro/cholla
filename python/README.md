# Overview

This directory primarily defines the `cholla_utils` package (more detail is provided below). With that said, it also contains some scripts more broadly related to working with Cholla.

For context some basic context:
- the **../pyproject.toml** file describes the ``cholla_utils`` package
- the **./cholla_utils** subdirectory contains the files in the actual package
- the **./contrib** subdirectory contains python logic that was previously **contrib**uted to the cholla repository, but is not actively maintained (in fact, some logic may not even work)
- the **./examples** subdirectory contains some examples of how to work with cholla datasets.
- the **./scripts** subdirectory contains some useful python scripts (that make use of the ``cholla_utils`` package)
- the **./tests** subdirectory contains some tests for the ``cholla_utils`` package

# `cholla_utils`

`cholla_utils` is a python package defining utilities functions and tools for working with Cholla outputs.

> [!NOTE]  
> Avoid executing any tests or scripts that import ``cholla_utils`` from within this directory (the presence of the ``cholla_utils`` subdirectory can confuse python).

<!-- SPHINX-START -->

## Installation

Currently, you **MUST** install this package from source.[^1] You can skip the rest of this section if you are already familiar with installation of python packages.

There are 3 ways to do this:

1. The first scenario is one where you haven't cloned the cholla repository, and don't intend to download it. In this case, you should invoke:

   ```sh
   python -m pip install --user git+https://github.com/cholla-hydro/cholla@dev
   ```

   Behind the scenes, pip will download the full repository, install the python package, and delete the repository (for context, the `@dev` at the end of the URL tells pip to install the package from the `dev` branch, rather than from the default `main` branch).

2. The next scenario is the case where you have cloned the cholla repositry and want to freeze the installed package. In other words, the `cholla_utils` package won't be affected by any modifications to python files in the cholla repository (that you intentionally make or that incidentally occur while switching between git branches), unless you perform a fresh installation of `cholla_utils`.

   In this case, invoke the following command from the root of the cholla repository (in other, invoke the command from the directory containing the **pyproject.toml** file):

   ```sh
   python -m pip install --user .
   ```

3. The final scenario is one where you have cloned the cholla repository and want to install `cholla_utils` in editable-mode. In other words, any (intentional or incidental) modifications within a python files will take effect the next time you freshly import from `cholla_utils`.[^2]

   In this scenario, you should the following command from the root of the cholla repository (in other, invoke the command from the directory containing the **pyproject.toml** file):

   ```sh
   python -m pip install --user -e .
   ```

## Quickstart - Field Data

Let's walk through a few scenarios.
Suppose that we have a snapshot saved at **path/to/snap.h5.0**.

> **Important aside:**, all of these examples will work if:
> - **path/to/snap.h5.0** contains the entire concatenated snapshot
>   - For the sake of clarity, the standard naming convention is to drop the suffix following **.h5** when you concatenate a file (in other words, the path to a concatenated file typically has the format **path/to/snap.h5**).
>   - The functionality we will discuss whether the file was concatenated with the newer version of the concatenation scripts (i.e. the resulting file retains the original block-structure) or the older version (or if the snapshot data was repacked)
> - OR, if **path/to/snap.h5.0** is one of many files that the snapshot is distributed between (i.e. there is no concatenation).
>   - This is the default format that Cholla writes, and the data is split between files that share the same path, except for the suffix after **.h5**
>   - In this case, it's important that you use the path variant that ends in **.0**.

Onto the examples:

1. If you want to query the fields that are saved within the dataset, you can invoke the following snippet:

   ```python
   import cholla_utils

   my_fields = cholla_utils.get_native_fields("path/to/snap.h5.0")
   print(my_fields)  # prints out the names of all saved fields
   ```

2. Loading a single field: This is easy. If that snapshot has a field called `"density"`, you can simply invoke:

   ```python
   import cholla_utils

   density = cholla_utils.load_field("path/to/snap.h5.0", "density")
   ```

   In the above snippet, the `density` variable holds a numpy array.

   Imagine that we are only interested in working with a subset of the array. For concreteness, imagine that we only want to work with `density[15,:,5:-20]`. The following snippet shows how we can directly load that subset of the data:

   ```python
   import cholla_utils

   density = cholla_utils.load_field(
       "path/to/snap.h5.0", "density", idx=np.s_[15,:,5:-20]
   )
   ```

3. Loading multiple fields: if we want to load the `"density"` and `"Energy"` fields at the same time, we would invoke:


   ```python
   import cholla_utils

   data = cholla_utils.load_field("path/to/snap.h5.0", ["density", "Energy"])
   # data is a dict
   # data["density"] holds the density numpy array
   # data["Energy"] holds the Energy numpy array
   ```

   Once again, we can tell the load function to only load a subset of the data by passing the `idx` keyword argument.


## Quickstart - Particle Data

Let's walk through a few scenarios related to particle data.
For concreteness, suppose that the particle data for a snapshot is saved at **path/to/snap/7_particles.h5.0**.

> **Important aside:**, similar to the functionality for reading field data, it  all of these examples will work if:
> - **path/to/7_particles.h5.0** contains the entire concatenated snapshot
>   - For the sake of clarity, standard naming convention dictates that a snapshot holding concatenated data is typically called **7_particles.h5** rather than **7_particles.h5.0**
>   - Unlike with the field data, this functionality can **ONLY** read concatenated particle data created by modern concatenation scripts (i.e. the resulting file maintains details about the original block-structure).[^3]
> - OR, if **path/to/snap.h5.0** is one of many files that the snapshot's particle data is distributed between (i.e. there is no concatenation).
>   - This is the default format that Cholla writes, and the data is split between files that share the same path, except for the suffix after **.h5**
>   - In this case, it **is** important that you use the path variant that ends in **.0**.

Let's move onto the examples:

1. If you want to query the particle properties that are saved within the dataset, you can invoke the following snippet:

   ```python
   import cholla_utils

   prop_pairs = cholla_utils.get_native_ptype_properties(
       "path/to/7_particles.h5.0"
   )
   print(prop_pairs)  # prints out all particle-property pairs
   ```

   The result is a list of two-tuples of the form `(<ptype>, <prop>)`, where `<ptype>` specifies particle-type and `<prop>` specifies a property name.
   This obviously merits further elaboration:
   - the choice to identify quantities associated with particles by these pairs, rather than just by the property name alone, was made to future-proof the API (and borrows from yt-conventions).
   - At the time of writing, Cholla is only capable of modelling (at most) a single particle type in a simulation.
     However, it's not hard to imagine a future where we add support for multiple types of particles (e.g. star-particles, dark matter particles, tracer particles), that may have different sets of properties (e.g. we might care about tracking masses and creation times for star particles, but not for dark matter particles).
   - At the time of writing, Cholla's distributed files don't directly record a particle type.
     We adopt yt's conventions and refer to these particles as having the `"io"` type.
     Cholla's particle concatenation script allows the user to specify an arbitrary particle name, but it too defaults to the `"io"` datatype.
   - standard particle properties include `"particle_IDs"`, `"pos_x"`, `"pos_y"`, ...

2. Loading a single field: This is easy. If that snapshot has a property called `("io", "pos_x")`, you can simply invoke:

   ```python
   import cholla_utils

   pos_x = cholla_utils.load_particle(
       "path/to/7_particles.h5.0", ("io", "pos_x")
   )
   ```

   In the above snippet, the `pos_x` variable holds a 1D numpy array.

   Imagine that we are only interested in working with a subset of the particle data.
   For concreteness, imagine that we only want to consider the data associated with the block in the leftmost corner of the simulation.
   The following snippet illustrates how we might accomplish this:

   ```python
   import cholla_utils

   density = cholla_utils.load(
       "path/to/7_particles.h5.0", ("io", "pos_x"), block_idx=np.s_[0,0,0]
   )
   ```

   If we instead wanted all blocks along the right x boundary, we could swap instead pass `np.s_[-1,:,:]` as the `block_idx` keyword argument.

3. Loading multiple particle properties: if we want to load the `("io", "particle_IDs")` and `("io", "pos_x")` particle properties at the same time, we would invoke:


   ```python
   import cholla_utils

   pos_x = cholla_utils.load_particle(
       "path/to/7_particles.h5.0", [("io", "particle_IDs"), ("io", "pos_x")]
   )
   # data is a dict
   # data[("io", "particle_IDs")] holds the particle_IDs numpy array
   # data[("io", "pos_x")] holds the pos_x numpy array
   ```

   Once again, we can tell the load function to only load a subset of the data by passing the `block_idx` keyword argument.

## Helpful Tips For Analyzing Both Particles and Field Data

When Cholla writes snapshots, it typically writes field data to a file named something like `63.h5.0` and particle data to a file of `63_particles.h5.0`.
If the files following this naming convention, the package provides convenience functionality so that if `cholla_utils.load_particle` and `cholla_utils.get_native_ptype_properties` will automaticlly infer that the user wants to read `63_particles.h5.0` if the functions are passed `63.h5.0`.

Furthermore, this functionality also works if:
- the field data is concatenated and the particle data is distributed
- the field data is distributed and the particle data is concatenated
- the field data and the particle data are both concatenated.

[^1]: If there is interest, it would be trivial to upload this package to PyPI.
[^2]: If you create/destroy/rename any python files, you probably need to fully reinstall `cholla_utils`.
[^3]: Cholla historically provided legacy particle concatenation scripts that mirrored the behavior of the legacy field concatenation scripts.
      However, we have decided not to support this format because, to our knowledge, these scripts were never used to concatenate production datasets.
