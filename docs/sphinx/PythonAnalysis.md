# Analyzing Outputs & `cholla_utils`

There are multiple ways to analyze Cholla Outputs.

To do this, you essentially have 3 options:

* Use the [`yt`](https://yt-project.org/) python package
* Use the `cholla_utils` python package.
  This is lighter weight alternative to `yt` and is shipped alongside Cholla.
  More details are provided below.
* Manually load in hdf5 files yourself.
  While this is necessary for 1D and 2D simulations, we **strongly** discourage this for 3D snapshots.
  If we need to tweak the output format of Cholla snapshots, your analysis scripts may break if you are manually loading in data files yourself (the other options are designed for forwards/backwards compatibility)

## More about `cholla_utils`

As noted above, `cholla_utils` is lightweight python package that provides functions and tools for working with Cholla outputs.

:::{include} ../../python/README.md
:start-after: <!-- SPHINX-START -->
:::
