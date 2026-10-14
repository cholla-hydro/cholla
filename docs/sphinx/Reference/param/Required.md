% In the future, I think it would be better if we distributed these more
% effectively under section titles that are more descriptive

:::{par:parameter} nx

:Summary: number of grid cells along the x dimension
:Type: {par:typefmt}`int`
:Default: *None*
:::

---

:::{par:parameter} ny

:Summary: number of grid cells along the y dimension
:Type: {par:typefmt}`int`
:Default: *None*

In a 1D problem, this must be set to 1
:::

---

:::{par:parameter} nz

:Summary: number of grid cells along the z dimension
:Type: {par:typefmt}`int`
:Default: *None*

In 1D and 2D problems, this must be set to 1
:::

---

:::{par:parameter} xmin

:Summary: x direction lower boundary (in code units)
:Type: {par:typefmt}`float`
:Default: *None*
:::

---

:::{par:parameter} ymin

:Summary: y direction lower boundary (in code units)
:Type: {par:typefmt}`float`
:Default: *None*
:::

---

:::{par:parameter} zmin

:Summary: z direction lower boundary (in code units)
:Type: {par:typefmt}`float`
:Default: *None*
:::

---


:::{par:parameter} init

:Summary: Name of initial conditions.
:Type: {par:typefmt}`string`
:Default: *None*

The value is case-sensitive.

Current options include:
- ``"Constant"``
- ``"Sound_Wave"``
- ``"Square_Wave"``
- ``"Riemann"``
- ``"Shu_Osher"``
- ``"Blast_1D"``
- ``"KH"``
- ``"KH_res_ind"``
- ``"Rayleigh_Taylor"``
- ``"Gresho"``
- ``"Implosion_2D"``
- ``"Noh_2D"``
- ``"Noh_3D"``
- ``"Disk_2D"``
- ``"Disk_3D"``
- ``"Disk_3D_particles"``
- ``"Spherical_Overpressure_3D"``
- ``"Spherical_Overdensity_3D"``
- ``"Clouds"``
- ``"Uniform_Grid"``
- ``"Zeldovich_Pancake"``
- ``"Chemistry_Test"``
- ``"Read_Grid"``
- ``"Read_Grid_Cat"``

See {repository-file}`src/grid/initial_conditions.cpp` for more information about each option.
Sample input parameter files for many of these problems can be found in the {repository-dir}`examples` directory.
:::

---

:::{todo}

Port over the remaining parameters
:::
