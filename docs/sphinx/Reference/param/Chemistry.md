
::::{par:parameter} chemistry.kind

:Summary: *kind of chemistry solver to use*
:Type: {par:typefmt}`str`
:Default: *depends on whether you using the legacy cooling Makefile flags*

This parameter specifies the choice of chemistry cooling solver.
For backwards compatibility, we set the default based on the presence of Makefile parameters.
When no Makefile parameters are provided, this defaults to "none."

With the exception of "chemistry-gpu" and "grackle" (more on those in a moment), you can freely overwrite the default value at runtime.
The following table summarizes the available choices (and the conditions where they become defaults):

:::{list-table}
:widths: auto
:header-rows: 1

* - Name
  - Meaning
  - Macros that make this the default
* - "none"
  - No chemistry or cooling
  - N/A
* - "tabulated-cloudy"
  - path to data file used by "tabulated-cooling" solver
  - `COOLING_GPU` && `CLOUDY_COOL`
* - "piecewise-cie"
  - piecewise-parabolic fit to a collisional ionization equilibrium (CIE)
  - `COOLING_GPU` **without** `CLOUDY_COOL`
* - "piecewise-ti"
  - Analytic cooling/heating recipe that roughly matches the "TI" cooling runs shown in
 in [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
  - N/A
* - "chemistry-gpu"
  - Non-equilibrium Hydrogen/Helium chemical network with heating/cooling
  - `CHEMISTRY_GPU`
* - "grackle"
  - CPU-based non-equilibrium primordial chemistry with heating/cooling
  - `COOLING_GRACKLE`
:::

:::{important}
At the time of writing, if you want to use "chemistry-gpu," you *MUST* use the `CHEMISTRY_GPU` Makefile parameter.
Likewise, if you want to use "grackle", you *MUST* use the `COOLING_GRACKLE` Makefile parameter.
In both cases, we do not allow the choices to be overwritten.
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

Can be used to enable photoelectric-heating when {par:param}`chemistry.kind` is set to "tabulated-cloudy" or "piecewise-ti". It is an error to specify this for any other {par:param}`cooling.kind`.

:::

---

:::{par:parameter} chemistry.photoelectric_n_av_cgs

:Summary: Parameterizes photoelectric heating
:Type: {par:typefmt}`float`
:Default: 100.0

When {par:param}`chemistry.photoelectric_heating` is `true`, this parameter can be used to specify the average number-density in the domain in cgs units (which is used to compute impact of photoelectric heating).

It is an error to specify this parameter when {par:param}`chemistry.photoelectric_heating` is `false`.
