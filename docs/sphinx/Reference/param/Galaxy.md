
### Initial CGM Properties

:::{par:parameter} model.galaxy.cgm_init.profile_kind

:Summary: *The desired gas halo profile*
:Type: {par:typefmt}`string`
:Default: isentropic

This plays a similar role to {par:param}`model.galaxy.gas_disk.init.profile_kind` (but applies to the CGM, rather than the gas disk).
At this time, this parameter **MUST** be set to ``isentropic``.

When set to ``isentropic``, the halo gas initially has constant entropy.
A phase diagram of this gas would show that it all lies along a constant adiabat.
This adiabat is "anchored" such that the gas temperature is equal to {par:param}`model.galaxy.cgm_init.T_anchor` at the specified mass density (either {par:param}`model.galaxy.cgm_init.rho_anchor_cgs` OR {par:param}`model.galaxy.cgm_init.rho_anchor_Msun_per_kpc3`).
This anchor point gas properties arise at a spherical radius equal to {par:param}`model.galaxy.cgm_init.R_anchor_kpc`

:::

---

:::{par:parameter} model.galaxy.cgm_init.T_anchor

:Summary: *The temperature that "anchors" the normalization of the CGM properties.*
:Type: {par:typefmt}`string`
:Default: *None: must be provided*

:::

---

:::{par:parameter} model.galaxy.gas_disk.init.rho_anchor_cgs

:Summary: *Specifies a mass density that "anchors" the normalization of the CGM gas profile.*
:Type: {par:typefmt}`string`
:Default: *None*

It is an error to provide this parameter when {par:param}`model.galaxy.cgm_init.rho_anchor_Msun_per_kpc3` is provided.

:::

---

:::{par:parameter} model.galaxy.cgm_init.rho_anchor_Msun_per_kpc3

:Summary: *Specifies a mass density that "anchors" the normalization of the CGM gas profile.*
:Type: {par:typefmt}`string`
:Default: *None*

It is an error to provide this parameter when {par:param}`model.galaxy.cgm_init.rho_anchor_cgs` is provided.

:::


---

:::{par:parameter} model.galaxy.cgm_init.R_anchor_kpc

:Summary: *Specifies the spherical radius where the anchoring properties of the CGM gas profile occur.*
:Type: {par:typefmt}`string`
:Default: *None*

:::


### Gas Disk

:::{par:parameter} model.galaxy.gas_disk.mass_Msun

:Summary: *The total mass of the gas disk*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

Technically this is the mass of the disk before truncation (in other words, the normalization of the density profile is unaffected by the radius where the disk is truncated).

:::

---

:::{par:parameter} model.galaxy.gas_disk.scale_radius_kpc

:Summary: *The scale_radius of the gas disk.*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

This is the {math}`e`-folding length-scale for an exponential gas disk

:::

---

:::{par:parameter} model.galaxy.gas_disk.selfgrav_scale_height_estimate_kpc

:Summary: *The scale height of the disk potential that is used as a rough guess for the self-gravity potential of the gas disk.*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

We essentially estimate the self-gravity potential of the gas disk as a "double exponential disk" {math}`\rho = \rho_0 \exp(R/R_d) \exp(|z|/h_z)`, where this parameter specifies {math}`h_z`.
In more detail, this potential is actually modelled as the superposition of 3 Miyamoto-Nagai disks.
The properties of these disks are inferred according to [Smith+ 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract).

The estimate for the self-gravity potential is used to help initialize circular-velocity in the initial conditions and it serves as the estimate for the potential at domain boundaries in the Paris-Galactic gravity solver.

:::

---

:::{par:parameter} model.galaxy.gas_disk.init.initial_scale_height_guess_kpc

:Summary: *Initial guess for the scale height when initializing the gas disk.*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

The actual scale height is iteratively computed.

:::

---

:::{par:parameter} model.galaxy.gas_disk.init.profile_kind

:Summary: *The kind of gas profile to use for initializing the gas disk.*
:Type: {par:typefmt}`string`
:Default: *None: must be provided*

The gas distribution for a gravitationally bound stable hydrodynamic system is usually computed given a relationship that specifies the thermal pressure as a function of mass density.
An analytic relationship commonly used for this purpose is called a polytrope or {math}`p=K \rho^\Gamma`, where {math}`\Gamma` is the polytropic index.

At the time of writing, this parameter should be set to ``isotropic``, which represents the special case with {math}`\Gamma=1` and the gas is all initialized at a constant temperature.
A user-provided temperature, {par:param}`model.galaxy.gas_disk.init.T_anchor`, is used to specify the normalization, when coupled with the equation of state for the gas.

In principle, this could be passed ``isentropic``, which corresponds to the special case with {math}`\Gamma=\gamma` (i.e. {math}`\gamma` is the adiabatic index).
In this case, all gas would be initialized with values along a constant adiabat (i.e. entropy is constant).
The adiabat with be determined by a user-specified temperature ({par:param}`model.galaxy.gas_disk.init.T_anchor`) and a user-specified mass-density ({par:param}`model.galaxy.gas_disk.init.rho_anchor_cgs` OR {par:param}`model.galaxy.gas_disk.init.rho_anchor_Msun_per_kpc3`).
In practice, this is a legacy choice that has **NOT** been rigorously tested.

:::

---

:::{par:parameter} model.galaxy.gas_disk.init.T_anchor

:Summary: *The temperature that "anchors" the normalization of the gas disk properties.*
:Type: {par:typefmt}`string`
:Default: *None: must be provided*

When {par:param}`model.galaxy.gas_disk.init.profile_kind` is ``isotropic`` all gas in the disk will be initialized with this temperature.
:::

---

:::{par:parameter} model.galaxy.gas_disk.init.rho_anchor_cgs

:Summary: *Specifies a mass density that "anchors" the normalization of the (non-isothermal) polytropic gas disk profile.*
:Type: {par:typefmt}`string`
:Default: *None*

It is an error to provide this parameter when {par:param}`model.galaxy.gas_disk.init.profile_kind` is ``isotropic`` or when {par:param}`model.galaxy.gas_disk.init.rho_anchor_Msun_per_kpc3` is provided.

:::

---

:::{par:parameter} model.galaxy.gas_disk.init.rho_anchor_Msun_per_kpc3

:Summary: *Specifies a mass density that "anchors" the normalization of the (non-isothermal) polytropic gas disk profile.*
:Type: {par:typefmt}`string`
:Default: *None*

It is an error to provide this parameter when {par:param}`model.galaxy.gas_disk.init.profile_kind` is ``isotropic`` or when {par:param}`model.galaxy.gas_disk.init.rho_anchor_cgs` is provided.

:::

### Star Forming Disk

:::{par:parameter} model.galaxy.star_forming_disk.global_sfr_Msun_per_kyr

:Summary: *Specfies the disk-wide integrated star formation rate*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

Setting this to 0 will prevent the creation of cluster particles.
This rate applies to the whole disk as if it were not truncated.

:::

---

:::{par:parameter} model.galaxy.star_forming_disk.poisson_point_process

:Summary: *Specfies the disk-wide integrated star formation rate*
:Type: {par:typefmt}`bool`
:Default: *None: must be provided*

When ``true`` clusters are created using a poisson point-process.
This means that all clusters can form simultaneously and, depending on sampling, the exact star formation rate may fluctuate with time.

Otherwise, clusters are formed one-at-a-time to ensure that the star formation rate is a constant value.
:::

% :::{par:parameter} model.galaxy.star_forming_disk.kennicut_schmidt_power
% 
% :Summary: *ADD ME*
% :Type: {par:typefmt}`floating-point`
% :Default: 1.4
% 
% :::

:::{note}
ADD REMAINING PARAMETERS
:::

### Static Potential

:::{par:parameter} model.galaxy.static_potential.halo.mass_Msun

:Summary: *The virial mass of the static halo potential*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

:::

---


:::{par:parameter} model.galaxy.static_potential.halo.concentration

:Summary: *The concentration parameter for the NFW profile*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

This is the ratio between scale the virial radius and the "scale radius," {math}`c = R_s / R_{\rm vir}`.

For added context, the "scale radius" directly parametrizes the NFW density profile

::::{math}
\rho(r)\equiv \rho_0 \frac{R_s}{r} \left(1+ \frac{r}{R_s}\right)^{-2}.
::::

:::

% Uncommenting the next line is problematic
% (is it because of the preceeding equation?)
%---

:::{par:parameter} model.galaxy.static_potential.halo.virial_radius_kpc

:Summary: *The virial radius of the static halo potential*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

:::

---

:::{par:parameter} model.galaxy.static_potential.old_stellar_disk.mass_Msun

:Summary: *Mass for static potential of the old (i.e. non-star forming) stellar disk*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

:::

---


:::{par:parameter} model.galaxy.static_potential.old_stellar_disk.scale_radius_kpc

:Summary: *Scale radius for the Miyamoto-Nagai static potential parameterization of the old (i.e. non-star forming) stellar disk*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

:::

---


:::{par:parameter} model.galaxy.static_potential.old_stellar_disk.scale_height_kpc

:Summary: *Scale height for the Miyamoto-Nagai static potential parameterization of the old (i.e. non-star forming) stellar disk*
:Type: {par:typefmt}`floating-point`
:Default: *None: must be provided*

:::
