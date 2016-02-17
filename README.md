CHOLLA
============
A 3D GPU-based hydrodynamics code

Getting started
----------------
This is the stable branch of the Cholla code. Cholla is designed to 
be run using NVIDIA GPUs, and can be run in serial mode using one GPU
or with MPI.

Two example makefiles are included in this directory, one designed for
linux and one for mac. After downloading the code, you should
be able to configure it for your machine by modifying one of the makefiles appropriately.


Configuration Notes
------------
Most of the options available in Cholla are selected within the makefile, e.g. single vs
double precision, output format, the reconstruction method, Riemann solver, and integrator.

A few options must be specified within on the 'FLAGS' line, namely the H_CORRECTION, dual energy (DE),
CUDA, and CUDA_ERROR_CHECK options. It is strongly recommended that you include the dual energy
flag when radiative cooling is turned on.

Running Cholla
--------------
To run the code after it is compiled, you must supply an input file with parameters and a problem that matches a function
in the initial_conditions file. E.g. to run a Sod Shock Tube test in one-dimension, you would do:

./cholla tests/1D/Sod.txt

Some output will be generated in the terminal, and output files will be written in the directory specified
in the input parameter file.

To run Cholla in parallel mode, you must un-comment the CHOLLA_MPI flag in the makefile. Then you can run
using:

mpirun -np 4 ./cholla tests/1D/Sod.txt

Each process will assign itself to a single GPU. Cholla cannot be run with more processes than available GPUs,
so MPI mode is generally most useful on a cluster, or for testing with a single process.


Other Things to Be Aware Of
--------------

In addition to selecting the CTU or VL integrators, the user
can also comment out both, in which case the integration scheme will revert to a simpler scheme in which 
the interface states and fluxes are calculated for each direction only once, and there are no transverse flux
corrections. In tests, this method often proves the most robust (and is also the fastest).

Cholla *can* be run without GPUs, but this configuration is not recommended. Because Cholla
was designed with GPUs in mind, the CPU performance is, at best, lackluster. In addition, many 
of the configuration options are not available in the non-CUDA mode (and I make no promises about whether 
the user is warned of this).

When running tests in fewer than 3 dimensions, cholla assumes that the x-direction will be used first, then
the y, then z. This is to say, in 1D nx must always be greater than 1, and in 2D nx and ny must be greater than 1.

Currently, Cholla is not designed to be able to run very large (>1e5) 1 dimensional problems. It's on the list.
