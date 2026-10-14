# Gravity
## Overview

Cholla Primarily implements 2 gravity-related modules:
- [Standalone Static Gravity](#static-gravity)
- [Self-Gravity](#self-gravity-fft-based)

:::{note}
At the time of writing, these 2 modules are presently incompatible with each another.
If you want to want to consider a static potential with the self-gravity solver, you currently need to write separate machinery for the self-gravity solver.
Honestly, this seems to be a historical artifact that would be straight-forward to fix (most other simulations codes treat static and self-gravity separately)
:::

Historically, Cholla also supported a SOR-based Self-Gravity, but that has not been maintained for a long time (and may be removed in the future)

## Static Gravity

Static gravity is activated using the ``STATIC_GRAV`` macro, and is used in several of the example problems provided with Cholla.
Static gravity is a simple prescription that does not require any other gravity flags, but does require the input file parameter ``custom_grav" to specify the analytic function that will be applied (dev branch only).
Static gravity is applied as momentum and energy source terms in [src/hydro/hydro_cuda.cu](https://github.com/cholla-hydro/cholla/blob/dev/src/hydro/hydro_cuda.cu) and the analytic functions are defined in [src/gravity/static_grav.h](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/static_grav.h).
As of 10-27-2023 on the main branch, the static gravitational field is hard-coded to provide a Milky Way-like model or an M82-like model. On the dev branch, the input parameter flags correspond to:

**1D:**
* 1: a MW-like Miyamoto-Nagai disk + NFW halo potential (assumed z = 0)

**2D:**
* 1: Gresho vortex
* 2: Rayleigh-Taylor instability
* 3: A 2D Keplerin disk
* 4: A MW-like Kuzmin disk + NFW halo potential (assumed z = 0)

**3D:**
* 1: A MW-like Miyamoto-Nagai disk + NFW halo potential
* 2: An M82-like Miyamoto-Nagai disk + NFW halo potential




## Self Gravity: FFT-based

In addition to static gravity, Cholla has an FFT-based self gravity solver.
Only one or the other may be used.
The self-gravity solver is turned on with the ``GRAVITY`` macro in the makefile.
The default behavior in the {repository-file}`config/make.type.gravity} build (and builds that depend on it) is also to turn on the ```GRAVITY_GPU``` macro, which ensures that gravity fields reside on the GPU (required for gpu-based MPI communications), and the ```PARIS``` macro, which specifies that the Poisson solve will be carried out on the GPU by the cuFFT or rocFFT libraries.
Cholla does also have CPU-based gravity solvers, although they are not currently maintained.
Definitions of other macros options associated with the gravity solver are given below.

In general, this module relies on a particle-mesh scheme. Broadly speaking, for each timestep Cholla:
- constructs a total density field that includes contributions of all gravitating dynamic mass. This typically include the mass density of gas and the mass from particles.[^1]
   - this density field **NEVER** include mass from a static background gravitational potential 
- passes this total density field to the Gravity solver to compute the associated Gravitational Potential
- subsequently uses this information (and potentially information about a static background potential) to account for gravitational forces

## Macro flags associated with self-gravity:

``GRAVITY``: Turns on self-gravity. Necessary for particle-only simulations.

``GRAVITY_GPU``: Specifies that fields required by gravity are allocated on the GPU.

``PARIS``: Use the Paris 3D GPU-based Poisson solver to calculate the gravitational potential on a periodic domain.

``GRAVITY_5_POINTS_GRADIENT``: Use a 5-point stencil to calculate the gradient of the potential for gravity source terms (default behavior is a 3-point stencil)

``GRAVITY_ANALYTIC_COMP``: Add an analytic component to the gravitational potential. As of 10-27-2023, this is hard-coded to a Milky Way galaxy model in the function `Setup_Analytic_Potential` from [gravity_functions.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/gravity_functions.cpp). 

``PARIS_3PT``: Use a 3-point gradient for the divergence operator approximation in Paris (default behavior is to use a spectral method)

``PARIS_5PT``: Use a 5-point gradient for the divergence operator approximation in Paris

``PARIS_GALACTIC``: Use the Paris Poisson solver on a domain with analytic boundaries set to match the selected model in the DiskGalaxy class.
As of 10-27-2023, this is hard-coded to a Milky Way galaxy model in the function `Compute_Gravitational_Potential` from [gravity_functions.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/gravity_functions.cpp) and in `Compute_Potential_Isolated_Boundary` from [gravity_boundaries.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/gravity_boundaries.cpp).

``PARIS_GALACTIC_3PT``: Same as above but for the analytic boundary version

``PARIS_GALACTIC_5PT``: Same as above but for the analytic boundary version

### Runtime Parameters associated with Static Gravity

At the time of writing, the only runtime parameter is {par:param}`gravity.gas_only_use_static_grav`


### Boundary Conditions: Context (and the trivial-case)

*General Background:* Boundary conditions are a messy part of any gravity solver. In general, the choice to use a method that solves gravity in Real space or Fourier space introduces complexity for handling periodic or non-periodic boundaries. As a general rule of thumb:
- real-space gravity solvers (e.g. tree-methods/multipole-methods) handle non-periodic boundaries trivially.
  They commonly resort to using Ewald summation for periodic boundaries.
- Because Fourier-space fundamentally represents real-space fields in terms of periodic basis functions, Fourier-space solvers find the scenarios with periodic boundaries to be relatively trivial.
  Non-periodic boundaries are much more tricky to handle.

Cholla's FFT-solver very much definitely adheres to this "rule of thumb." Consequently, it trivially handles simulations with periodic boundaries (e.g. cosmological simulations).
We discuss the non-periodic case below.

## Non-Periodic Boundary Strategy

Before digging in, we briefly introduce some (fairly standard) notation:  {math}`{\bf x}` denotes a spatial position, {math}`\rho({\bf x})` is a mass density field in the simulation (in this particular discussion, we include contributions from gas **AND** particles), {math}`{\bf g}({\bf x})` is the gravitational acceleration, and {math}`\phi({\bf x})` is the gravitational potential

### The Strategy's Ingredients
Our solution for the non-periodic case involves 2 ingredients: a modified scheme for the Poisson-solve and a static estimate for the potential produced by {math}`\rho_{\rm tot}`

1) **Modified scheme for Poisson-Solve:** This scheme makes use of Discrete Sine Transforms (implemented in terms of FFT machinery).
Instead of requiring periodic boundaries, this variant requires the following boundary conditions:
    1. {math}`{\bf g}({\bf x}_{\rm bound}) = {\bf 0}`, or equivalently {math}`{\bf \nabla} \phi \rvert_{{\bf x}_{\rm bound}} = {\bf 0}`
    2. {math}`\rho({\bf x}_{\rm bound}) = 0`, or equivalently {math}`\nabla^2\phi\rvert_{{\bf x}_{\rm bound}} = 0`
    While this modified-scheme no longer requires periodic boundary conditions, it clearly is **NOT** the full solution (only pathological scenarios can satisfy condition 1).

2) **A static estimate for the gravitational potential produced by {math}`\rho_{\rm tot}`**: In more detail, we actually need a static estimate for the density-potential pair
   - Added context: in case you aren't familiar with the concept, every valid gravitational potential profile is associated with a unique density profile. Thus we describe the pair of profiles as a "density-potential" pair [^density-potential-pair]
   - Let's call these quantities {math}`\rho_{\rm estimate}({\bf x})` and {math}`\phi_{\rm estimate}({\bf x})`
   - At the time of writing, the solver requires {math}`\rho_{\rm estimate}({\bf x})` and {math}`\phi_{\rm estimate}({\bf x})` to have analytic formulae. (However, the implementation could be generalized to use numerically computed profiles)

### Tying things together:

Let's briefly review the "inputs" and "outputs":
- at startup, the gravity solver is initialized so that it "knows" {math}`\rho_{\rm estimate}({\bf x})` and {math}`\phi_{\rm estimate}({\bf x})`
- during each timestep Cholla provides the solver with the density-field of the dynamical mass {math}`\rho_{\rm tot}({\bf x})` and expects it to output the associated potential {math}`\phi_{\rm tot}({\bf x})`

We now describe the solver's procedure:
1. Compute {math}`\rho_{\rm offset}({\bf x})=\rho_{\rm tot}({\bf x}) - \rho_{\rm estimate}({\bf x})`
2. Use the Poisson-solve to compute the gravitational potential {math}`\phi_{\rm offset}\left({\bf x}\right)` from {math}`\rho_{\rm offset}\left({\bf x}\right)`.
3. Compute {math}`\phi_{\rm tot}({\bf x})=\phi_{\rm offset}({\bf x}) + \phi_{\rm estimate}({\bf x})`

This procedure implicitly leverages 2 fundamental concepts. First, density-potential pairs are additive (e.g. if you have 2 point-masses, you can add together their potentials to get the total gravitational potential). Second, You might notice here that {math}`\rho_{\rm offset}({\bf x})` can have negative values. While the concept of negative density is unintuitive (since negative mass is somewhat meaningless), there aren't any mathematical issues with computing gravitational potential from negative potentials.[^negative-density]

It's instructive to revisit boundary conditions for the Poisson-solve and consider them in the context of the procedure's second step:
1. {math}`{\bf g}_{\rm offset}({\bf x}_{\rm bound}) = {\bf \nabla} \phi_{\rm offset} \rvert_{{\bf x}_{\rm bound}} =  {\bf \nabla} \phi_{\rm tot} \rvert_{{\bf x}_{\rm bound}} - {\bf \nabla} \phi_{\rm estimate} \rvert_{{\bf x}_{\rm bound}}`
2. {math}`\rho_{\rm offset}({\bf x}_{\rm bound})=\rho_{\rm tot}({\bf x}_{\rm bound}) - \rho_{\rm estimate}({\bf x}_{\rm bound})`

This places important requirements on {math}`\rho_{\rm estimate}({\bf x})` and {math}`\phi_{\rm estimate}({\bf x})`:
- the boundary requirement of the Poisson-solve are satisfied as long as BOTH {math}`{\bf \nabla} \phi_{\rm estimate}` is a good estimate for {math}`{\bf \nabla} \phi_{\rm tot}` at the boundaries **AND**  {math}`\rho_{\rm estimate}` is a good estimate for {math}`\rho_{\rm tot}` at the boundaries
- as long as the above conditions  are met, it is okay for {math}`\rho_{\rm estimate}({\bf x})` to have large deviations from {math}`\rho_{\rm tot}({\bf x})`.
  This can only happen far from the boundaries.

:::{warning}
If {math}`{\bf \nabla} \phi_{\rm estimate}` isn't a good estimate for {math}`{\bf \nabla} \phi_{\rm tot}` at the boundaries **OR**  {math}`\rho_{\rm estimate}` isn't a good estimate for {math}`\rho_{\rm tot}`, then the artifacts and inaccuracies will be introduced into the solution
:::


## Support for Problem-types with Non-Periodic Boundaries

Currently, the only scenario that Cholla supports with non-periodic boundaries is idealized Galaxy simulations.

At the time of writing, current density-potential pairs used in these problems have some **minor** "flaws:"
- they don't account for the gravitational potential from halo gas
- they don't account for the potential of star particles
- they make use of density-potential pairs that don't account for disk-truncation

Of these 3 "flaws," the third is probably most troubling (since the estimated density-potential pair models contributions from material beyond the edges of the simulation domain). However, these flaws are probably not a concern since everything seems qualitatively reasonable (it's hard to quantify this).

[^1]: In more detail, we use the fairly standard second-order cloud-in-cell (CIC) interpolation technique to distribute the particle mass onto the field. This total density field is the primary input for this solver.

[^negative-density]: If you are still struggling to wrap your head around this, it may be instructive to recall the parallels with electrostatics. In that context, Poisson's equation, with physical different constants, relates electrostatic potential and charge density (which can be negative).

[^density-potential-pair]: For the uninitiated, a given density profile always has a unique gravitational potential (modulo a constant offset in the potential).  An example of a well-known potential pairs is the point mass.
  As we all know, the density profile of a point mass (of mass {math}`M`) is associated with {math}`\phi(r) = -GM/r`.
  Another example is the Plummer-Kuzmin disk potential (where {math}`R` is cylindrical radius).
  In this case, an infinitely thin disk with surface density profile {math}`\Sigma(R) = a M/ (2\pi (R^2 + a^2)^{3/2})` has the potential {math}`\phi(R,z) = -GM/\sqrt{R^2 + (a + |z|)^2}`.
