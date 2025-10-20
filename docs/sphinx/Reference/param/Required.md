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

:::{todo}

Port over the remaining parameters
:::
