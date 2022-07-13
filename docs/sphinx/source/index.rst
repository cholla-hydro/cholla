.. Cholla documentation master file, created by
   sphinx-quickstart on Wed Mar 31 19:47:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Cholla documentation
===================================

Cholla is a static-mesh, GPU-native hydrodynamics simulation code that efficiently runs high-resolution simulations on massively-parallel computers. The code is written in a combination of C++ and Cuda C and requires at least one NVIDIA GPU to run. Cholla was designed for astrophysics simulations, and the current release includes the following physics:

  * compressible hydrodynamics in 1, 2, or 3 dimensions
  * optically-thin radiative cooling and photoionization heating, including the option to use the Grackle cooling library
  * static gravity with user-defined functions
  * FFT-based gas self-gravity
  * particle-mesh based particle gravity
  * cosmology
  * passive scalar tracking

Cholla can be run using a variety of different numerical algorithms, allowing users to test the sensitivity of their results to the exact code configuration. Options include:

  * Exact, Roe, and HLLC Riemann solvers
  * 2nd and 3rd order spatial reconstruction with limiting in either primitive or conserved variables
  * CTU or Van Leer integrators

Please cite the original code paper (Schneider & Robertson, 2015, ApJS) if you use Cholla for your research. Additional pages in this wiki describe how to set up and run the code. Happy simulating!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gettingstarted
   input
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
