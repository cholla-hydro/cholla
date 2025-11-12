
:::{par:parameter} feedback.boundary_strategy

:Summary: *Specify handling in the scenario when a feedback event is scheduled to occur for a particle near the edge of the grid-block that contains the particle and the stencil overlap with cells outside of the grid-block (this includes ghost cells).*
:Type: {par:typefmt}`str`
:Default: *None: must be provided*

Valid options include:

- `"ignore_issues"`: as its name implies, this choice ignores the issue (essentially we lose some fraction of the injected "stuff").
- `"snap"`: compute the stencils as if the source-particle positions were snapped to the closest position in the grid-block such that the stencil only includes cells within a block.

In the future, a more robust strategy will involve MPI communication (and/or ghost particles)

:::

---

:::{par:parameter} feedback.snr_filename

:Summary: path to the table used to determine the supernova rate (for the `"table"` rates)
:Type:    {par:typefmt}`str`
:Default: *None*

This parameter is meaningless if {par:param}`feedback.sn_rate` isn't set to `"table"`.

If this parameter is not set, then a default constant SNR is used.
The default SNR corresponds to 1 supernova per {math}`100 M_\odot` of cluster mass, spread out over 36 Myr, starting when the cluster is 4 Myr old.
A sample Starburst99 file is included in the source code at `src/particles/starburst99_snr.txt`.
The sample represents a {math}`10^6 M_\odot` fixed mass cluster, created using a Kroupa initial mass function, and with an {math}`8 \mathrm{M}_\odot` supernova cutoff.
More details are provided {ref}`here. <general-SNe-rate>`

:::

---

:::{par:parameter} feedback.sn_model

:Summary: Specifies the name of the supernova model
:Type: {par:typefmt}`str`
:Default: *None*

More details are provided {ref}`here <SNe-Prescription-Descriptions>`.

:::

---

:::{par:parameter} feedback.sn_rate

:Summary: Specifies the kind of supernova rate
:Type: {par:typefmt}`str`
:Default: `"table"`

When `"table"` (the default value) is specified, Cholla determines the rate from a table.
`"immediate_sn"` schedules a single supernova to occur, immediately after the simulation starts.
More details are provided ({ref}`here <general-SNe-rate>`
