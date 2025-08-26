# Cholla

:::{warning}
This website is under construction.
We're currently in the process of transferring contents from the [Cholla Wiki](https://github.com/cholla-hydro/cholla/wiki) to this website
:::

<!-- Here we list contents (that populate the sidebare). -->

:::{toctree}
:maxdepth: 1
:hidden:

Introduction <self>
GettingStarted
CompilingCholla
PythonAnalysis
Physics/index
ChollaExamples/index
PythonExamples/index
Reference/index
:::

:::{toctree}
:caption: About the Project
:hidden:

Development/index
Citation
Help
:::


:::{toctree}
:caption: Useful links
:hidden:

GitHub <https://github.com/cholla-hydro/cholla>
Issue Tracker <https://github.com/cholla-hydro/cholla/issues>
:::

<!-- Down below we describe information for the landing page of the website.
     Once we start hosting the site, we should really deduplicate most of
     the contents between this site and the README.md file at the root of
     the repository -->

Cholla is a static-mesh, GPU-native magnetohydrodynamics (MHD) simulation code that efficiently runs high-resolution simulations on massively-parallel computers. The code is written in a combination of C++, Cuda C, and HIP, and requires at least one NVIDIA or AMD GPU to run. Cholla was designed for astrophysics simulations, and the current release (3.0.0) includes the following physics:
* compressible hydrodynamics or ideal MHD in 1, 2, or 3 dimensions
* optically-thin radiative cooling and photoionization heating, including the option to use Cloudy cooling tables or the [Grackle](https://grackle.readthedocs.io/en/grackle-3.1/) cooling library
* static gravity with user-defined functions
* FFT-based gas self-gravity
* particle-mesh based particle gravity
* cosmology
* passive scalar tracking

Cholla can be run using a variety of different numerical algorithms, allowing users to test the sensitivity of their results to the exact code configuration. Options include:
* Exact, Roe, HLLC, and HLLD Riemann solvers
* 2nd and 3rd order spatial reconstruction with limiting in either primitive or conserved variables
* a second order Van Leer integrator

Please cite the original code paper (Schneider & Robertson, 2015, ApJS) if you use Cholla for your research. Please also cite the cosmology paper (Villasenor et al. 2021, ApJ) if you use self-gravity or cosmology. Please cite the MHD paper (Caddy & Schneider 2024, ApJ) if you use MHD. Additional pages in this wiki describe how to set up and run the code. Happy simulating!
