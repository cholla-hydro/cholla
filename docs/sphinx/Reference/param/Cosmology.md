:::{par:parameter} H0

:Summary: Present-day Hubble Rate in units of km/s/Mpc
:Type: {par:typefmt}`float`
:Default: *None*
:::

---

:::{par:parameter} Omega_M

:Summary: Present-day matter energy-density
:Type: {par:typefmt}`float`
:Default: *None*
:::

---

:::{par:parameter} Omega_L

:Summary: Present-day dark energy energy-density
:Type: {par:typefmt}`float`
:Default: *None*
:::

---

:::{par:parameter} Omega_b

:Summary: Present-day baryonic matter energy-density
:Type: {par:typefmt}`float`
:Default: *None*
:::

---

:::{par:parameter} Omega_R

:Summary: Present-day radiation energy-density
:Type: {par:typefmt}`float`
:Default: 0.0
:::

---

:::{par:parameter} w0

:Summary: Present-day dark energy equation of state. The {math}`w_0` in a {math}`w(a) = w_0 + w_a (1-a)` dark energy equation of state parameterization.
:Type: {par:typefmt}`float`
:Default: -1.0
:::

---

:::{par:parameter} wa

:Summary: Linear interpolation of dark energy equation of state to early Universe. The {math}`w_a` in a {math}`w(a) = w_0 + w_a (1-a)` dark energy equation of state parameterization.
:Type: {par:typefmt}`float`
:Default: 0.0
:::

---

:::{par:parameter} scale_outputs_file

:Summary: Path to data file describing the scale factor values to output snapshots.
:Type: {par:typefmt}`str`
:Default: *None*

We expect increasing scale factor values in this file
:::

---

:::{par:parameter} wDE_file

:Summary: Path to data file describing a redshift-dependent dark energy equation of state
:Type: {par:typefmt}`str`
:Default: *None*

We expect increasing redshift values in this file. Each row must have a `z wDE(z)` value with a space delimiter. Table must have an entry at redshift {math}`z=0`. Table should look like

```shell-session
# output the first five lines of wDE_file
$ head wDE_file -n 5
# z, w
0.000000000000000000e+00 -9.767616499273475972e-01
6.938631476027579126e-03 -9.769941793369097960e-01
1.392540755881421788e-02 -9.772263520514015145e-01
2.096066230604587410e-02 -9.774581369813842846e-01
```


When specified, overrides {par:param}`w0` and {par:param}`wa`.
:::

---

:::{todo}

include Init_redshift
:::
