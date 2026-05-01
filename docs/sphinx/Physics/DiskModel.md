# Disk Model
## Overview

One of Cholla's flagship applications is Idealized Galaxy Simulations of disk galaxies.
These simulations can involve choices related to several physical concepts: (i) modeling of gravity, (ii) radiative cooling, and (iii) modeling stellar feedback. These modules are all highly related.
You also need to consider 2 simulation phases: initial-conditions and evolution over a timestep.

### Primer on Gravitational Potentials

A central ingredient of any galaxy simulation is the physical model of the gravitational potential. This is a complex topic that is relevant in the context of initial conditions **AND** in the context of evolving the simulation over a timestep. The precise treatment of the potential depends on the context **AND** on Cholla's configuration.

It is instructive to first define all possible components of the total potential. Throughout the rest of this page, we will subsequently describe how treatment of these potentials can vary. The total potential is given by: 

:::{math}

\Phi_{\rm tot} = \Phi_{\rm dm} + \Phi_{\rm stars,old} + \Phi_{\rm stars,young} + \Phi_{\rm gas}
:::

where
-  {math}`\Phi_{\rm dm}` corresponds to the dark matter halo  (always a static potential)
-  {math}`\Phi_{\rm stars,old}` corresponds to older stellar populations in the disk (always a static potential)
- {math}`\Phi_{\rm stars,young}` corresponds to the younger stellar populations (computed with self-gravity from particles)
- {math}`\Phi_{\rm gas}` corresponds to the gas disk (computed with self-gravity from particles)

### Primer on Radiative Cooling

At the time of writing, simulations must always be initialized with CIE cooling (cooling shuts off below 1e4 K) in order to prevent immediate collapse of the gas disk.

To use other cooling mechanisms, we currently recommend that you:

- start running the simulation with CIE cooling

- after enough time has passed to start driving turbulence in the gas disk (commonly a few 10s of Myr), you can then restart the simulation with a different cooling recipe.

### Primer on Feedback and Star Formation:

We provide a basic particle-based feedback approach.
At the moment, all star particles must be initialized at startup (see below).
In the future, the goal is to introduce self-consistent star formation.

## Configurations

Historically, there have been 3 main physics-module configurations for running an Idealized Galaxy Problem:
1. Standalone Static Gravity:
  - In this mode, standalone static gravity is used and particles are **NOT** supported.
  - Effectively, this mode assumes that {math}`\Phi_{\rm stars,young}=0` and {math}`\Phi_{\rm gas}=0`.
  - **Purpose:** A simpler model that is easy to customize
:::{warning}
This configuration has not been tested for quite a while in the dev branch.
:::

:::{caution}
At the time of writing, the logic modelling the static potential is duplicated between the initial conditions and the source-term calculation.
It is the user's responsibility to ensure that consistent models of the potential are used in both parts of the code.
:::

2. Normal Self Gravity
  - use the self-gravity module to model the full gravitational potential while updating timesteps.
  - We provide details about assumptions that the gravity-solver makes to compute contributions from self-gravity [down below](#analytic-self-gravity-estimate)
  - approximations for the total potential are used during initial conditions
  - **Purpose:** to model the disk of a galaxy as self-consistently as possible (it's currently unclear whether collapsing gas clouds in simulations without self-consistent star formation are an issue)
3. Particle-Self-Gravity
  - uses the self-gravity module to effectively achieve a hybrid between the other 2 modes.
  - enable this mode by setting the `gravity.gas_only_use_static_grav` runtime parameter to `true`
  - The gravitational forces are handled differently for star particles and gas:
    - gravity source terms for the gas are only computed from the static potentials: {math}`(\Phi_{\rm dm} + \Phi_{\rm stars,old})`.
    - the star particles experience forces from all components {math}`(\Phi_{\rm dm} + \Phi_{\rm stars,old} + \Phi_{\rm stars,young} + \Phi_{\rm gas})`.
    - This merits some emphasis: star particles do indeed experience gravity from the gas (as well as from themselves) while the gas does **not** experience its own self-gravity. This choice is motivated by the assumptions we make for the gravity-solver to compute contributions from self-gravity [down below](#analytic-self-gravity-estimate)
  - **Purpose:** this is intended to be an intermediate point between both of the other modes that lets us confidently experiment with a 3-phase ISM without self-consistent star formation.







## Analytic Gravitational Potentials

In this section, we discuss the analytic formulae used for modeling Gravitational Potentials. It may be instructive to look back at our [definitions of various components](#primer-on-gravitational-potentials).
### Static Potentials

First, let's consider static potentials. 

All configurations of the idealized Galaxy Simulations:
- currently assume that you are using a static Gravitational Potential produced by 2 components: (i) component contributed by older stellar populations {math}`\Phi_{\rm stars-old}` and (ii) a component contributed by dark matter in the halo, {math}`\Phi_{\rm halo}`.
- use these potentials for initializations and for evolving the simulation over a timestep.

:::{important}
[As we noted above](#configurations), when using the standalone Static-Gravity module, the logic implementing the static potential is duplicated between the initial conditions and the source-term calculations; it's the user's responsibility to ensure logic remains consistent in both parts of the code. 
This is **NOT** a concern when using the Self-Gravity module
:::

Consider the parameterization of a Miyamoto-Nagai potential (this will be important [again](#analytic-self-gravity-estimate), shortly)

:::{math}
\Phi_{\rm MN}(R,z; M,a,b)\equiv \frac{-G M}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}.
:::

At the time of writing, we always define {math}`\Phi_{\rm stars,old}(R,z)=\Phi_{\rm MN}(R,z; M_{\rm stars}, R_{\rm stars}, z_{\rm stars})`, where {math}`R` & {math}`z` are cylindrical coordinates, {math}`R_{\rm stars}` & {math}`z_{\rm stars}` are the scale radius/height, and {math}`M_{\rm stars}` is the mass of the disk.
Currently, these values are hardcoded where we initialize the `ClusteredDiskGalaxy` struct that holds the Milky Way properties (in {repository-file}`src/model/particles/disk_galaxy.cu`).

At the time of writing, {math}`\Phi_{\rm dm}` is always assumed to be an NFW profile, or

:::{math}
\Phi_{\rm dm}(r) = \frac{-G M_{\rm vir}}{r[\ln(1+c) - c/(1+c)]}\ \ln \left(1 +\frac{r}{R_{\rm dm}}\right),
:::

where {math}`r=\sqrt{R^2+z^2}` is radius in spherical coordinates, {math}`M_{\rm vir}` is the dark matter mass, {math}`c` is halo concentration, and {math}`R_{\rm dm}` is the scale length of the halo.


### Analytic Self-Gravity Estimate

Now we turn our attention to an analytic estimate related to self-gravity. Be aware: when updating a simulation over a timestep, the gravitational forces acting on particles and the fluid are **NEVER** directly computed from this formula.

Specifically, we define {math}`\hat{\Phi}_{\rm gas,disk}(R,z)`. As we will [become clear below](#gas-initial-conditions), we can roughly approximate our gas disk is roughly characterized by a double-exponential surface density profile 

:::{math}
\rho_{\rm d}(R,) = \rho_{{\rm d},0} \exp(-R/R_{\rm d}) \exp(-|z|/z_{\rm d}).
:::

The potential for this density profile does **NOT** have an analytic form. Instead, we use the tables from [Smith+15](https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S) to define {math}`\hat{\Phi}_{\rm gas}(R,z)` as the superposition of 3 Miyamoto-Nagai disk potentials that approximate the disk potential.

This approximation comes up in 2 cases:
- During initialization, we assume that {math}`\hat{\Phi}_{\rm gas,disk}(R,z)` is a good approximation for the self gravity of gas disk and star particles when computing circular velocities.
- In the context of the self-gravity solver, we assume that {math}`\hat{\Phi}_{\rm gas,disk}(R,z)` is a good approximation for the total potential {math}`(\Phi_{\rm gas,disk}(R,z) + \Phi_{\rm gas,halo}(R,z) + \Phi_{\rm stars,young}(R,z))`  at the boundaries of the simulation domain. As discussed on the [[Gravity]] page of the documentation, if this is a bad approximation, then artifacts will arise in the simulation

The primary limitation to this approximation for {math}`\hat{\Phi}_{\rm gas,disk}(R,z)` is that we don't account for the fact that the gas disk is truncated. Truncation has 2 impacts:
- less importantly: the shape of truncation can be important for the shape of the potential near the truncation radius
- more importantly: {math}`\hat{\Phi}_{\rm gas,disk}(R,z)` includes the contributions to the gravitational potential from material outside of the simulation domain (if that sounds "wrong" to you, that's becomes most of your physical intuition comes from spherical symmetry).

## Gas Initial Conditions

The ``"Disk_3D"`` and ``"Disk_3D_particles"`` initial conditions will both initialize the gas disk. Both cases initialize an isothermal gas disk within a halo. The latter also subsequently initializes particles.

Cholla offers machinery to setup an idealized galaxy simulation. Specifically, it initializes properties a gas disk inside of gas halo. The conditions broadly match the description provided by  [Schneider & Robertson 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S) (there are some extensions when using self-gravity).

Ideally, our initial conditions would be in hydrostatic equilibrium (i.e. the properties of a simulation would not change if we hit "go"). For a handful of reasons, we can't actually achieve perfectly hydrostatic initial conditions (related to the gas in the halo and the boundary conditions), but we pick ICs that are stable.

To accomplish this, it is useful to consider gas in the disk and gas in the halo. Let  {math}`\rho_{\rm disk}(R, z)` and {math}`P_{\rm disk}(R,z)` refer to the axis-symmetric profiles for the disk gas. We define {math}`\rho_{\rm halo}(r)` and {math}`P_{\rm halo}(r)` as the spherical profiles for gas in the halo. We will provide precise definitions for these profiles momentarily. To ensure a smooth transition between disk and halo, we define the simulation's initial density and pressure profiles as 

:::{math}
:nowrap:

\begin{eqnarray}
\rho(R,z) = \rho_{\rm disk}(R, z) + \rho_{\rm halo}(\sqrt{R^2+z^2}) \\
P(R,z) = P_{\rm disk}(R, z) + P_{\rm halo}(\sqrt{R^2+z^2}).
\end{eqnarray}
:::

It's instructive to note that while {math}`\rho_{\rm disk}` & {math}`P_{\rm disk}` are both 0 "outside of the disk," {math}`\rho_{\rm halo}` & {math}`P_{\rm halo}` are **always** positive.

At all spatial locations within the gas disk (i.e. where {math}`\rho_{\rm disk}>0`), we introduce a rotational velocity, {math}`v_{\rm rot}`, to support the disk or that

:::{math}

\frac{v_{\rm rot}^2 (R, z)}{R} = -\frac{\partial\Phi_{\rm cyl,est}}{\partial R} + \frac{1}{\rho(R,z)} \frac{\partial P}{\partial R},
:::

where {math}`\Phi_{\rm cyl,est}` is our best estimate of the total potential (this is defined below).

:::{note}
Earlier versions of Cholla **_only_** considered the density and pressure of the disk's gas when it computed {math}`v_{\rm rot}` (in other words, it didn't consider gas from the halo).
:::

### A quick primer on Solving Density Profiles in Hydrostatic Eq

It is instructive to briefly consider the generic solution for density profiles in Hydrostatic Equilibrium. In 1D hydrostatic equlibrium requires that:

:::{math}

\frac{1}{\rho} \frac{\partial P}{\partial x} + \frac{\partial \Phi}{\partial x} = 0.

:::

In spherical symmetry, we can replace {math}`x` with {math}`r`. If you are considering vertical stratification of a planar "atmosphere", you can replace {math}`x` with {math}`z`.

In all cases we assume {math}`\Phi` is independent of {math}`\rho`. To derive a solution, we need an equation of state (EoS) relating {math}`P` and {math}`\rho`. Solutions are typically derived using the the generic polytropic equation of state: {math}`P = K\rho^\Gamma `, where
- {math}`K` is a constant 
- {math}`\Gamma` is a constant called the "polytropic index" (sometimes "polytropic index" refers to a separate, related quantity). The EoS is isothermal (temperature is constant) for {math}`\Gamma=1` and isentropic (entropy is constant) for {math}`\Gamma=\gamma`.
It is instructive to note that {math}`c_s^2 = \partial P / \partial \rho = K \Gamma \rho^{\Gamma - 1} `

The solution is:

:::{math}
:nowrap:

\begin{eqnarray}
\rho(x) &=& \rho(x_0) \exp\left(-\frac{\Phi(x) - \Phi(x_0)}{c_s^2}\right)&,&\ \Gamma = 1 \\
\rho(x) &=& \rho(x_0) \left[ 1+(\gamma - 1) \frac{\Phi(x) - \Phi(x_0)}{c_s^2(x_0)}\right]^{1/(\gamma-1)}&,&\ {\rm otherwise}
\end{eqnarray}

:::

### Halo Gas

Lets consider the halo gas profile.

As noted in the [Schneider & Robertson 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S) , "in setting the halo gas distribution the potential is taken to be spherically symmetric." In other words, the halo gas profile uses a potential called {math}`\Phi_{\rm sph}(r) = \Phi_{\rm stars}(R=0,z=r) + \Phi_{\rm halo}(r)` (it **NEVER** includes contributions from self-gravity). We use an isentropic profile:

:::{math}

\rho_{\rm h}(r) = \rho_{\rm h}(r_0)\left[ 1+(\gamma - 1) \frac{\Phi_{\rm sph}(r) - \Phi_{\rm sph}(r_0)}{c_{s,{\rm h}}^2(r_0)}\right]^{1/(\gamma-1)}.

:::

In the above equation
- {math}`r_0` is the "cooling radius." This effectively is where we choose to normalize the profile. Essentially, the parameterized by specifying {math}`\rho_{\rm h}(r_0)` .
- {math}`c_{s,{\rm h}}^2(r_0)` is the adiabatic sound speed at the cooling radius, or {math}`c_{s,{\rm h}}(r_0)= \sqrt{\gamma k_B T_{\rm h}(r_0) / (\mu m_{\rm H})}`.

:::{note}
Earlier versions of the code historically defined {math}`c_{s,{\rm h}}` as the isothermal sound speed. Typically, this means that profile never reached the associated temperature by {math}`r_0`.
:::

### Gas Disk Profile

The gas disk is initialized with an isothermal profile. An exponential gas disk surface density profile is {math}`\Sigma(R) = \Sigma_0 \exp(-R / R_{\rm d})`. In practice, we truncate the surface density. The modified surface density equation is given by:

:::{math}

\Sigma(R) = \Sigma_0 \exp(-R / R_{\rm d}) (1- f(r)),
:::

where {math}`f(R)` is a modified form of logistic function or 

:::{math}

f(R) = 0.5 + 0.5 \tanh\left(\frac{R -R_{\rm trunc}}{2\alpha}\right).
:::

At this time {math}`R_{\rm trunc}` is computed based on the domain, and {math}`\alpha` is hardcoded (they should probably be runtime parameters).

:::{note}
Earlier versions accomplished tapering in a slightly different manner
:::

For fixed {math}`R`, we can solve for a given hydrostatic column through the disk. The solution is governed by the following pair of equations:

:::{math}
:nowrap:

\begin{eqnarray}
\Sigma(R) &=& 2\int_{0}^\infty\rho_{\rm disk} (R, z)dz \\
\rho_{\rm disk}(R, z) &=& \rho_{\rm disk} (R, z=0)\ \exp\left(-\frac{\Phi_{\rm vert}(R,z) - \Phi_{\rm vert}(R,z=0)}{c_s^2}\right)
\end{eqnarray}
:::

Recall that the latter equation [is the solution for an isothermal fluid in Hydrostatic equilibrium](#a-quick-primer-on-solving-density-profiles-in-hydrostatic-eq).

#### Ignoring Gas Self-Gravity

When ignoring gas self gravity, these equations are easy to solve since {math}`\Phi_{\rm vert}` is just the analytic equation: {math}`\Phi_{\rm vert}=\Phi_{\rm dm} + \Phi_{\rm stars,old}`. These equations become:

:::{math}
:nowrap:

\begin{eqnarray}
\rho_{\rm disk}(R, z=0) &=&  \frac{\Sigma(R)}{2} \left[\int_{0}^\infty\exp\left(-\frac{\Phi_{\rm vert}(R,z) - \Phi_{\rm vert}(R,z=0)}{c_s^2}\right)dz\right]^{-1} \\
\rho_{\rm disk}(R, z) &=& \rho_{\rm disk} (R, z=0)\ \exp\left(-\frac{\Phi_{\rm vert}(R,z) - \Phi_{\rm vert}(R,z=0)}{c_s^2}\right)
\end{eqnarray}
:::

#### Including Disk-Gas Self-Gravity

This case is a little more complex. In this case, {math}`\Phi_{\rm vert} = \Phi_{\rm gas, disk} + \Phi_{\rm dm} + \Phi_{\rm stars,old}`. We describe the {math}`\Phi_{\rm gas, disk}` with the simplified form of Poisson's Equation for an axisymmetric thin disc or

:::{math}

\frac{d^2\Phi_{\rm gas, disk}}{dz^2} = 4 \pi G \rho_{\rm gas,disk}.
:::

For more details about why its okay to ignore neglect variations along {math}`R`, see the discussion of equation 14 and Appendix E in [Wang+ 2010](https://ui.adsabs.harvard.edu/abs/2010MNRAS.407..705W/abstract). For convenience, we define {math}`\Phi_{\rm gas, disk}(R,z=0) = 0` (we can do this since we only think about a single {math}`R` at a time).

Let's also define the variable {math}`u` as:

:::{math}

u(R, z)=\int_{0}^z\frac{\rho_{\rm disk} (R, z)}{\rho_{\rm disk} (R, 0)} dz = \int_{0}^z \exp\left(-\frac{\Phi_{\rm vert}(R,z) - \Phi_{\rm vert}(R,z=0)}{c_s^2}\right) dz.
:::

In other words, {math}`\rho_{\rm disk} (R, 0) = \Sigma(R)/(2 u(R,\infty))`.

**What does that do for us?** We can recast our various equations as:

:::{math}
:nowrap:

\begin{eqnarray}
\frac{d}{dz} \frac{d\Phi_{\rm gas, disk}}{dz} &=& 4 \pi \rho_{\rm disk} (R, 0)\ \exp\left(-\frac{\Phi_{\rm gas, disk}(R,z) + (\Phi_{\rm dm}(R,z)-\Phi_{\rm dm}(R,0)) + (\Phi_{\rm stars,old}(R, z) - \Phi_{\rm stars,old}(R, 0))}{c_s^2}\right) \\
\frac{d}{dz} \Phi_{\rm gas, disk} &=& \frac{d}{dz} \frac{d\Phi_{\rm gas, disk}}{dz} \\
\frac{d}{dz} u &=& \exp\left(-\frac{\Phi_{\rm gas, disk}(R,z) + (\Phi_{\rm dm}(R,z)-\Phi_{\rm dm}(R,0)) + (\Phi_{\rm stars,old}(R, z) - \Phi_{\rm stars,old}(R, 0))}{c_s^2}\right)
\end{eqnarray}
:::

We have written the equations in this form to highlight that this is an "initial value problem." To solve for the {math}`\Phi_{\rm gas, disk}(R,z)` and the density profile (again, this is all at fixed {math}`R`), we follow this procedure:
1. Start with an initial guess for {math}`\rho_{\rm disk} (R, 0)`
2. Use our guess for {math}`\rho_{\rm disk} (R, 0)` to integrate the system of 3 differential equations from {math}`z = 0` out to a "large z" (where {math}`\rho_{\rm disk}` is essentially 0).
   - we know that {math}`\frac{d\Phi_{\rm gas, disk}}{dz}(R,z=0) = 0` (since it's an extrema), {math}`\Phi_{\rm gas, disk}(R,z=0)=0` (we picked this earlier) and {math}`u(R,z=0)=0` (by definition)
   - this directly gives us {math}`\frac{d\Phi_{\rm gas, disk}}{dz}(R,z)`, {math}`\Phi_{\rm gas, disk}(R,z)`, and {math}`u(R,z)`
3. We compute a new guess for {math}`\rho_{\rm disk} (R, 0)` from {math}`\Sigma(R) / (2 * u(R, \infty))`
4. If our new guess was "close enough" to our previous guess, we're done. Otherwise, go back to step 2 using our newest guess.

## Particle ICs

Particles are only initialized if you use the ``"Disk_3D_particles"`` initial conditions.

We use the Kennicutt–Schmidt law to determine the distribution of particles with respect to {math}`r_{\rm cyl}`

:::{math}

\Sigma_{SFR}(R) = a * \Sigma_{\rm gas}^{k_s},
:::

where `a` is some arbitrary normalization constant and {math}`k_s` is usually 1.4. We can combine this with the formula for the gas surface density, {math}`\Sigma_{\rm gas}(R) = \Sigma_{{\rm gas},0} \exp(-R / R_{\rm d})`, to get a more detailed formula for star-formation surface density:

:::{math}

\Sigma_{\rm SFR}(R) = a * \Sigma_{{\rm gas},0}^{k_s} * \exp(-r_{\rm cyl} / R_{\rm gas-scale-length}).
:::

Essentially we can use this to incremental rate of star-formation {math}`d{\rm SFR}` in the disk between {math}`r_{\rm cyl}` and {math}`(r_{\rm cyl} + dr_{\rm cyl}`). If we consider some duration of time (in practice set by the `t_out` runtime parameter), you can work an expected number of stars formed between  {math}`r_{\rm cyl}` and {math}`(r_{\rm cyl} + dr_{\rm cyl}`) during that duration. It's straight-forward to convert this function into a PDF.

:::{note}
I can definitely elaborate more -- I found the relevant notes on this topic
:::

At startup we sample this PDF to determine the {math}`r_{\rm cyl}` at which all particles are expected to form. We create star-particles at these radii and distribute the "turn-on times" from a time a little before we start the simulation until the time specified by the `tout` parameter. We use Poisson sampling to distribute these "turn-on times," to target a particular SFR.
At the time of writing, the SFR is hardcoded within the `disk_stellar_cluster_init_` C++ function (in {repository-file}`src/particles/particles_3D.cpp`).

:::{todo}
Adjust the location where SFR is set to be co-located with other parameters.
(Ideally, it would be a runtime parameter)
:::



