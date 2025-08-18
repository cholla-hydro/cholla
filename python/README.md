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

## Quickstart

Let's walk through a few scenarios. Suppose that we have a snapshot saved at **path/to/snap.h5.0**. *Importantly*, all of these examples will work if:
- **path/to/snap.h5.0** contains the entire concatenated snapshot (In this scenario, the convention is to name the file **path/to/snap.h5**). It doesn't matter whether the file was concatenated with the newer or older scripts (or whether it was repacked)
- OR, if **path/to/snap.h5.0** is one of many files that the snapshot is distributed between (i.e. there is no concatenation)

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

   density = cholla_utils.load("path/to/snap.h5.0", "density")
   ```

   In the above snippet, density holds a numpy array.

   Imagine that we are only interested in working with a subset of the array. For concreteness, imagine that we only want to work with `density[15,:,5:-20]`. The following snippet shows how we can directly load that subset of the data:

   ```python
   import cholla_utils

   density = cholla_utils.load(
       "path/to/snap.h5.0", "density", idx=np.s_[15,:,5:-20]
   )
   ```

3. Loading multiple fields: if we want to load the `"density"` and `"Energy"` fields at the same time, we would invoke:


   ```python
   import cholla_utils

   data = cholla_utils.load("path/to/snap.h5.0", ["density", "Energy"])
   # data is a dict
   # data["density"] holds the density numpy array
   # data["Energy"] holds the Energy numpy array
   ```

   Once again, we can tell the load function to only load a subset of the data by passing the `idx` keyword argument.

[^1]: If there is interest, it would be trivial to upload this package to PyPI.
[^2]: If you create/destroy/rename any python files, you probably need to fully reinstall `cholla_utils`.
