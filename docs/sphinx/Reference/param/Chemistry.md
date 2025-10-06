
:::{par:parameter} chemistry.kind

:Summary: *kind of chemistry solver to use*
:Type: {par:typefmt}`string`
:Default: *depends on whether you using the legacy cooling Makefile flags*

This parameter specifies the choice of chemistry cooling solver.

Valid Options include:
- "none": (don't use cooling)
- "tabulated-cloudy"
- "piecewise-cie"

:::

---

:::{par:parameter} chemistry.data_file

:Summary: path to data file used by "tabulated-cooling" solver
:Type:    {par:typefmt}`str`
:Default: *an empty string*

It is an error to specify this file when {par:param}`chemistry.kind` chemistry cooling solver other than "tabulated-cooling."

:::

---

add more...
