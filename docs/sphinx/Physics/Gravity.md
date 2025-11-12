# Static Gravity

Static gravity is activated using the ```STATIC_GRAV``` macro, and is used in several of the example problems provided with Cholla. Static gravity is a simple prescription that does not require any other gravity flags, but does require the input file parameter "custom_grav" to specify the analytic function that will be applied (dev branch only). Static gravity is applied as momentum and energy source terms in [src/hydro/hydro_cuda.cu](https://github.com/cholla-hydro/cholla/blob/dev/src/hydro/hydro_cuda.cu) and the analytic functions are defined in [src/gravity/static_grav.h](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/static_grav.h). As of 10-27-2023 on the main branch, the static gravitational field is hard-coded to provide a Milky Way-like model or an M82-like model. On the dev branch, the input parameter flags correspond to:

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

In addition to static gravity, Cholla has an FFT-based self gravity solver. Only one or the other may be used. The self-gravity solver is turned on with the ```GRAVITY``` macro in the makefile. The default behavior in the [make.type.gravity](https://github.com/cholla-hydro/cholla/blob/dev/builds/make.type.gravity) build (and builds that depend on it) is also to turn on the ```GRAVITY_GPU``` macro, which ensures that gravity fields reside on the GPU (required for gpu-based MPI communications), and the ```PARIS``` macro, which specifies that the Poisson solve will be carried out on the GPU by the cuFFT or rocFFT libraries. Cholla does also have CPU-based gravity solvers, although they are not currently maintained. Definitions of other macros options associated with the gravity solver are given below.

Macro flags associated with self-gravity:

```GRAVITY```: Turns on self-gravity. Necessary for particle-only simulations.

```GRAVITY_GPU```: Specifies that fields required by gravity are allocated on the GPU.

```PARIS```: Use the Paris 3D GPU-based Poisson solver to calculate the gravitational potential on a periodic domain.

```GRAVITY_5_POINTS_GRADIENT```: Use a 5-point stencil to calculate the gradient of the potential for gravity source terms (default behavior is a 3-point stencil)

```GRAVITY_ANALYTIC_COMP```: Add an analytic component to the gravitational potential. As of 10-27-2023, this is hard-coded to a Milky Way galaxy model in the function `Setup_Analytic_Potential` from [gravity_functions.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/gravity_functions.cpp). 

```PARIS_3PT```: Use a 3-point gradient for the divergence operator approximation in Paris (default behavior is to use a spectral method)

```PARIS_5PT```: Use a 5-point gradient for the divergence operator approximation in Paris

```PARIS_GALACTIC```: Use the Paris Poisson solver on a domain with analytic boundaries set to match the selected model in the DiskGalaxy class. As of 10-27-2023, this is hard-coded to a Milky Way galaxy model in the function `Compute_Gravitational_Potential` from [gravity_functions.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/gravity_functions.cpp) and in `Compute_Potential_Isolated_Boundary` from [gravity_boundaries.cpp](https://github.com/cholla-hydro/cholla/blob/dev/src/gravity/gravity_boundaries.cpp).

```PARIS_GALACTIC_3PT```: Same as above but for the analytic boundary version

```PARIS_GALACTIC_5PT```: Same as above but for the analytic boundary version

```PARIS_GALACTIC_TEST```: Turn on to test whether Paris returns the same gravitational potential as the SOR solver. Doesn't work with GRAVITY_GPU, should probably be deprecated.

## Self Gravity: SOR based
To-do: Describe the SOR solver
