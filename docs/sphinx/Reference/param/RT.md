
:::{par:parameter} rt_num_iterations

:Summary: maximum number of radiative transfer substeps per hydro step
:Type: {par:typefmt}`int`
:Default: `10`

Used when compiled with `RT_M1` or `RT_OTVET`.

With `RT_M1`, a hydro step of length {math}`\Delta t` on cells of size {math}`\Delta x` is split into {math}`n` substeps, where {math}`N` is `rt_num_iterations` and the RT CFL number is 0.5:

- legacy scheme ({par:param}`rt_reduced_c` {math}`\le 0`): {math}`n = \min\left(1 + \lceil c\,\Delta t / (0.5\,\Delta x) \rceil,\; N\right)`, and light advances {math}`1/n` of a cell per substep.
- reduced speed of light ({par:param}`rt_reduced_c` {math}`= f > 0`): {math}`n = \min\left(\max(1, \lceil f c\,\Delta t / (0.5\,\Delta x) \rceil),\; N\right)`, and light advances {math}`\min(f c\,\Delta t / (n\,\Delta x),\; 0.5)` cells per substep. Light can therefore travel at most {math}`0.5\,N` cells per hydro step.

With `RT_OTVET`, the number of iterations per hydro step is {math}`\min\left(N,\; 1 + c\,\Delta t / \Delta x\right)`.

Larger values let the radiation respond on shorter time scales, at the cost of more RT kernel launches per hydro step.

:::

---

::::{par:parameter} rt_reduced_c

:Summary: reduced speed of light factor for the M1 scheme
:Type: {par:typefmt}`float`
:Default: `0` (legacy scheme)

Used when compiled with `RT_M1`. It sets how far light travels per hydro step; see {par:param}`rt_num_iterations` for the substep formulas.

:::{list-table}
:widths: auto
:header-rows: 1

* - Value
  - Meaning
* - `0` (or negative)
  - Legacy scheme: light advances one cell per hydro step, whatever the hydro time step, so the effective speed of light is {math}`\Delta x / \Delta t`. The non-cosmological Iliev tests are validated with this setting.
* - `> 0`
  - Reduced speed of light {math}`c_\mathrm{eff} = f\,c` with {math}`f` = `rt_reduced_c`. When {math}`f c\,\Delta t / \Delta x` exceeds {math}`0.5 \times` {par:param}`rt_num_iterations`, light travels only that many cells per step, and {math}`c_\mathrm{eff}` is lower than requested.
:::

Each hydro step the code prints the effective speed of light as `RT: c_eff/c = ...`.

:::{important}
Runs compiled with `COSMOLOGY` should set `rt_reduced_c > 0`.
A cosmological hydro step usually spans many light-crossing times of a cell (about {math}`10^3` at {math}`z = 7` in the {ref}`3D-Shapiro-Giroux` test), so the legacy scheme would give {math}`c_\mathrm{eff} \sim c/10^3`.
That test uses `rt_reduced_c = 1`.
Make sure that the ionization fronts of interest move much more slowly than the printed {math}`c_\mathrm{eff}`.
:::

::::

---

:::{par:parameter} rt_outputs_file

:Summary: path to a file listing RT output times
:Type: {par:typefmt}`str`
:Default: *None*

Used when compiled with `RT_M1` or `RT_OTVET`, and ignored with `COSMOLOGY`, where snapshot times come from {par:param}`scale_outputs_file`.

The file lists one output time per line, in code time units (kyr).
Entries more than {math}`10^{-3}` before the start time are skipped.
Without this parameter, the RT output times are the initial time and then every `outstep` (scaled by `outstep_dexinc`, if set) until `tout`.

The run ends once the last RT output time is reached, and prints `RT: Reached Last RT Output: Ending Simulation`.

:::
