# Particle-based Feedback

This page covers particle-based feedback. Essentially particles are used to model star-clusters and periodically they are scheduled to inject mass/material/energy as feedback into a fluid field. When an event occurs, mass is removed from a star cluster. The cells of the simulated grid that are modified are known as the event's "stencil."

At this time, we only support supernovae. (Foundations for stellar-wind feedback can also be found in the codebase, but a significant work is needed to fully implement it).

## Enabling Particle-Based Feedback
At this time, the user needs to define the `FEEDBACK` macro (we can convert that to a runtime parameter with minimal effort).

The user is also responsible for defining macros for properly configuring the particle macro:
* `PARTICLES_GPU` - Particle-based feedback requires that the particle data be on the GPUs.
* `PARTICLE_AGE` - Feedback varies with cluster age.
* `PARTICLE_IDS` - Cluster IDs are used to prevent possible correlations or biases when generating random numbers used by the feedback algorithm.

## Runtime Parameters
At the time of writing, the runtime parameters for configuring feedback are:
- {par:param}`feedback.boundary_strategy`
- {par:param}`feedback.snr_filename`
- {par:param}`feedback.sn_model`
- {par:param}`feedback.sn_rate`

You can find a more complete description of these parameters {ref}`here <Reference-Feedback-Runtime-Params>`.

## Supernovae Rate 

This is controlled by the {par:param}`feedback.sn_rate` parameter.
This topic is split into 2 parts: (i) {ref}`general-rate <general-SNe-rate>` and (ii) [handling events from neighboring particles](#handling-events-from-neighboring-particles).

(general-SNe-rate)=
##  General Rate

Setting {par:param}`feedback.sn_rate` to `"immediate_sn"` is primarily used for testing-purposes.

This discussion assumes that you set the {par:param}`feedback.sn_rate` parameter to `"table"`. Our treatment of the SN rate is inspired by [Kim & Ostriker (17)](https://ui.adsabs.harvard.edu/abs/2017ApJ...846..133K/abstract). For clusters with a mean age bounded by {math}`[t_m, t_m+dt)`, we consider the specific supernovae rate (i.e. the number of supernovae, {math}`\mathcal{N}_{\rm SN}`, per cluster mass {math}`M_{\rm cl}`)
${math}`\xi_{\rm SN}(t_{\rm m}) = \frac{d}{dt}\left(\frac{\mathcal{N}_{\rm SN}}{M_{\rm cl}}\right).`$
For a star-cluster that currently has mass {math}`M_{\rm cl}^\prime` and mean age {math}`t_{\rm m}^\prime`, we draw the number of supernovae in the cluster over a global simulation timestep {math}`\delta t` from a Poisson distribution expected value {math}`(\delta t M_{\rm cl}^\prime\xi_{\rm SN}(t_{\rm m}^\prime))`.  This treatment
- means that a given cluster can have multiple SNe during a single timestep (this is reflected by our prescriptions)
- clearly makes some assumptions (e.g. {math}`\xi_{\rm SN}(t_{\rm m})` is roughly constant over {math}`\delta t`, changes in {math}`M_{\rm cl}` during a timestep doesn't dramatically impact the probability of subsequent events)

As in [Kim & Ostriker (17)](https://ui.adsabs.harvard.edu/abs/2017ApJ...846..133K/abstract), we use the table of results from `STARBURST99` for a fully sampled Kroupa IMF to infer {math}`\xi_{\rm SN}(t_{\rm m})`. In practice, Cholla reads in the `STARBURST99` results from the file specified by the {par:param}`feedback.snr_filename` parameter to pre-tabulates {math}`\xi_{\rm SN}(t_{\rm m})` at simulation startup.

:::{important}
We seed the PRNG to try to make a given simulation deterministic.
If you use exactly the same version of the code, in the exact same configuration, the supernova rate *should* remain consistent (useful in the context of restarts and debugging).
However, note that seeding depends on particle ids and the number of simulation timesteps (from t=0).
Additionally, the algorithms for sampling a Poisson distribution may vary between CUDA & HIP.
:::

### Handling events from neighboring particles

In this subsection, we address the question:

> "How do we handle the scenario when 2 or more particles are scheduled to undergo feedback during the same timestep and have overlapping stencils?" 

This is an **extremely** important question for a code where separate threads are applying feedback for separate particles to a single block in parallel (it's also relevant in any code if a single feedback stencil is allowed to modify cells in multiple blocks).  A naive implementation will lead to race-conditions.

**Our solution:** we "sequence" the supernova events within a single timestep based on particle id. Essentially when multiple particles have events that can modify common cells, we sequentially apply feedback (the order is dictated by particle id). This behavior is simple and deterministic, and importantly it's well-defined in arbitrarily complex scenarios (e.g consider a small cluster of 5 particle with overlapping stencils, but there are no cells that are overlapped by more than 3 stencils). 

:::{note}
While this "sequencing" solution isn't very "physical," it only makes a practical difference when stencils overlap.
If it is coming into play frequently in a given simulation, we may want to consider alternatives, like shorter timesteps or subcycling (after all, the probability that 2 nearby star clusters would have a supernova at exactly the same time is extremely low)
:::

The main alternative that we could revisit is implementing the method prescriptions as atomic operations. If we want to adopt this solution in the future, there are a number of important considerations that need to be addressed in the future. These details are highlighted in the next subsection.


#### ASIDE: Important considerations for atomic "conflict-resolution"

Essentially we walk through 3 increasingly complex scenarios:
1. **Pure thermal energy-injecting prescription:** if all SN events **_only_** inject energy this is easy to accomplish with atomics.
2. **Thermal Energy and Mass injection:** is more complex if only because kinetic energy density scales linearly with mass density while thermal energy density is independent of mass density.
   - it becomes more complex if you modify the gas momentum to account for the fact that you're injecting the mass in the particle's reference frame.
   - In this scenario, you need to (i) subtract off the kinetic from the entire total energy density field before anything else, (ii) atomically modify the fluid fields, (iii) add kinetic energy (using the new values) back to the total energy density field
1. **Thermal Energy, Momentum, and Mass injection:** in practice this isn't any more complex than the last case (the same strategy applies)
2. **Dynamically adjusting the prescription based on local conditions:** theoretically you can determine the kind of prescription each particle will use first, and then apply the same procedure that was used in the last 2 cases. In practice, this may not be sensible (e.g. if deciding between resolved vs unresolved)
3. **Prescriptions that can average values within stencils in addition to injections:** I don't think there is a robust, straight-forward way to make this work with atomics (that also scales to arbitrarily complex scenarios)

(SNe-Prescription-Descriptions)=
## Prescription

This section describe's the actual prescriptions. First we offer some broad background context. Then, we offer a high-level description of the available prescriptions. Subsequently, we provide additional detail about the stencils that are actually used and the magnitudes of the injected quantities.
### Background Context

Broadly speaking, it's useful to introduce the distinction between a "resolved" prescription that primarily injects thermal energy, and an "unresolved" prescription that primarily injects momentum. In slightly more detail:
- a simulation with infinite resolution would always use "resolved" prescriptions that injects thermal energy.
- Injecting thermal energy is problematic at coarser resolutions because overcooling prevents us from resolving a supernova remnant's evolution. Conventional wisdom holds that its better to simply inject the final momentum that we expect the remnant to produce after the remnant would have enough time to evolve to spatial scales that are large enough to be resolved. 
- both kinds of prescriptions may inject mass. Properly accounting for a star-particle's reference frame means that mass injection implicitly involves some level of modifying the momentum of the gas.

It is also possible to come up with other schemes that inject both thermal energy and momentum  (as in [Kim & Ostriker 17](https://ui.adsabs.harvard.edu/abs/2017ApJ...846..133K/abstract)).

:::{note}
In the context of Cholla, we adopt slightly more precise definitions of "resolved" and "unresolved."
:::

## High-Level Description of Prescriptions in Cholla:

At the time of writing, Cholla adopt the following definitions of "resolved" and "unresolved" feedback:
- resolved feedback always injects thermal energy and transfers mass. 
- unresolved feedback is 2-step prescription that first overrides the density and momentum for the full cells that have at least partial overlap with the stencil with the average value (across the region being overwritten). Then, the final momentum is injected and mass is transferred.
In both cases, we properly account for the particle's reference frame (when transferring mass and injecting momentum). We also propagate changes to total energy density.

Presently, Cholla supports pure resolved-feedback prescriptions and hybrid-prescriptions. In a hybrid prescription:
- we compute the total mass in all cells that at least partially overlap with the stencil divide it by the total volume of those cells to get an average mass density. We convert that to a number density {math}`n_0`.
- We can plug this into {math}`r_{\rm sf} = 22.6\, {\rm pc}\ N_{\rm SN}^{0.29} n_0^{-0.42}` to get the radius of shell formation. This comes from equation 8 [Kim & Ostriker (15)](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract) (the {math}`N_{\rm SN}^{0.29}`  term comes from replacing {math}`E_{51}^{0.29}`)
- When, {math}`r_{\rm sf} > 3\Delta x`, we use the resolved prescription. Otherwise, we use the unresolved prescription

Here we consider the values of {par:param}`feedback.sn_model`. There are currently a couple of flavors (some of them are for experimental purposes). Some flavors are just resolved, while others are hybrid prescriptions. The stencils also vary [(we describe the stencils below)](#stencil-descriptions).

- {par:param}`feedback.sn_model` = `"resolvedCiC"`
  - kind: resolved feedback
  - stencil: we use standard 8-cell Cloud-in-Cell Interpolation
- {par:param}`feedback.sn_model` = `"resolved27cell"`
  - kind: resolved feedback
  - stencil: we use a spherical 27-cell stencil, with supersampling to calculate overlap
  - **EXPERIMENTAL**
-  {par:param}`feedback.sn_model` = `"legacy"`
  - kind: hybrid
  - stencil: resolved feedback uses standard 8-cell Cloud-in-Cell (it's **exactly** like `"resolvedCiC"`).
    Unresolved feedback uses a legacy 27-cell stencil based on CiC Interpolation (i.e. the stencil is effectively a cube)
  - **IMPORTANT:** please see the description of the momentum stencil before you pick this choice.
- {par:param}`feedback.sn_model` = `"legacyAlt"`
  - kind: hybrid
  - stencil: - stencil: resolved feedback uses standard 8-cell Cloud-in-Cell (it's **exactly** like `"resolvedCiC"`). Unresolved feedback uses a spherical 27-cell stencil, with supersampling to calculate overlap
  - **EXPERIMENTAL:** (this is probably a better choice that `"legacy"` since the momentum-stencil is better defined)

:::{todo}
We may want to drop a few unneeded models, and rename some options
:::

### Stencil Descriptions

The stencil is tied to way we distribute the source terms among cells from a feedback event.
- as part of a stencil calculation, we might calculate the fraction of a cell's volume that is enclosed within the stencil-volume
- when it comes to mass, we usually try to inject a constant amount of mass (or thermal energy) per unit volume throughout the stencil's volume.
- momentum is trickier since it's a vector (with 3 components). In general, we try to (i) distribute momentum density as evenly as possible and (ii) ensure that we don't introduce any net momentum (in the reference frame of the source star-cluster). But there are some thorny questions (that we won't directly address):
  - "what do we do when there is cancellation?" (obviously less of an issue for a larger stencil)
  - "do we want the magnitude of the momentum vector to be constant?" Presumably, the answer is yes, but "what about non-spherical deposition regions?"

Types of Stencils:
- Cloud-in-Cell Interpolation:
  - essentially, we treat the deposition volume as cube with side-length {math}`\Delta x` (the volume is centered on the particle). For context, a sphere with radius {math}`\Delta x / 2` is about half the volume.
  - This is way too small for dealing with momentum
- Legacy-27 stencil based on CiC
  - in broad strokes, the idea is to "divid[e] a scalar quantity between 3x3x3 cells is done by imagining a 2x2x2 cell volume around the SN"
  - it's unclear how the vector deposition strategy was derived (the original author can't figure out how he arrived at the answer)
  - In this case, the deposition region is essentially a cube where each side has a width {math}`2\Delta x`
- spherical 27-cell stencil, with supersampling to calculate overlap
  - essentially, we treat the deposition volume as a sphere with radius  {math}`\Delta x`. We compute the fractional overlap of cells with the sphere using supersampling. 
  - momentum-deposition is based on estimating the integral of the radial unit-vector over a cell.

Future work may want to consider the use of larger stencils.
### Magnitude Description

Every supernova transfers {math}`10 M_\odot` from the star particle.

Resolved feedback injects  {math}`E_{\rm SN}=10^{51}\ {\rm erg}` of thermal energy

:::{NOTE}
ASIDE: In our simulations that inject {math}`2\times10^{51} {\rm erg}`, I'm pretty sure that I simply modified the hardcoded value.
Because of the way that the code was originally written I didn't realize that the equations for {math}`r_{\rm sf}` and {math}`p_{\rm final}` implicitly hid their injection-energy dependence.
Thus, in those simulation I don't think we accounted for that energy dependence.
:::

At the time of writing, unresolved prescriptions use a final momentum of {math}`p_{\rm final}=2.8\times 10^5 M_\odot\ {\rm km}\ {\rm s}^{-1} n_0^{-0.17} N_{\rm SN}^{0.93}`
-  everything but the {math}`N_{\rm SN}^{0.93}`  term comes from equation 34 of [Kim & Ostriker (15)](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract), which is direct fit to a number of different simulations of a supernova remnant  ( [Kim & Ostriker 17](https://ui.adsabs.harvard.edu/abs/2017ApJ...846..133K/abstract) also use this formula).
- the {math}`N_{\rm SN}^{0.93}`  term comes from the {math}`E_{51}^{0.93}` term in equation 17 of [Kim & Ostriker (15)](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract).
- In this equation, Cholla plugs the value of {math}`n_0` that was computed while determining if a supernova is resolved. I believe that Cholla's choice of mean molecular weight that differs from  [Kim & Ostriker 15](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract) and [Kim & Ostriker 17](https://ui.adsabs.harvard.edu/abs/2017ApJ...846..133K/abstract) 

