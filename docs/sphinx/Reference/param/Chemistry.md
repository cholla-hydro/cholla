
::::{par:parameter} chemistry.kind

:Summary: *kind of chemistry solver to use*
:Type: {par:typefmt}`str`
:Default: "none" (unless the `CHEMISTRY_GPU` or `COOLING_GRACKLE` macro is defined)

This parameter specifies the choice of chemistry cooling solver.
When `CHEMISTRY_GPU` and `COOLING_GRACKLE` are not defined, this defaults to "none."

:::{important}
At the time of writing, if you want to use "chemistry-gpu," you *MUST* use the `CHEMISTRY_GPU` Makefile parameter.
Likewise, if you want to use "grackle", you *MUST* use the `COOLING_GRACKLE` Makefile parameter.
In both cases, we do not allow the choices to be overwritten.
For user convenience, the default value are set to these options when the Makefile parameter is detected.
:::

:::{list-table}
:widths: auto
:header-rows: 1

* - Name
  - Meaning
* - "none"
  - No chemistry or cooling
* - "tabulated-cloudy"
  - path to data file used by "tabulated-cooling" solver
* - "piecewise-cie"
  - piecewise-parabolic fit to a collisional ionization equilibrium (CIE)
* - "piecewise-ti+cie"
  - Analytic cooling/heating recipe that roughly matches the "TI" cooling runs shown in
 in [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
* - "chemistry-gpu"
  - Non-equilibrium Hydrogen/Helium chemical network with heating/cooling (only usable when the `CHEMISTRY_GPU` macro is defined)
* - "grackle"
  - CPU-based non-equilibrium primordial chemistry with heating/cooling (only usable when the `COOLING_GRACKLE` macro is defined)
:::

::::

---

:::{par:parameter} chemistry.data_file

:Summary: path to data file used by "tabulated-cloudy" solver
:Type:    {par:typefmt}`str`
:Default: *None*

It is an error to specify this parameter when {par:param}`chemistry.kind` chemistry cooling solver other than "tabulated-cloudy."

:::

---

:::{par:parameter} chemistry.photoelectric_heating

:Summary: enables photoelectric-heating
:Type: {par:typefmt}`bool`
:Default: `false`

Can be used to enable photoelectric-heating when {par:param}`chemistry.kind` is set to "tabulated-cloudy" or "piecewise-ti+cie". It is an error to specify this for any other {par:param}`cooling.kind`.

:::

---

:::{par:parameter} chemistry.photoelectric_n_av_cgs

:Summary: Parameterizes photoelectric heating
:Type: {par:typefmt}`float`
:Default: 100.0

When {par:param}`chemistry.photoelectric_heating` is `true`, this parameter can be used to specify the average number-density in the domain in cgs units (which is used to compute impact of photoelectric heating).

It is an error to specify this parameter when {par:param}`chemistry.photoelectric_heating` is `false`.
