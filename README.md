CHOLLA
============
A 3D GPU-based hydrodynamics code (Schneider & Robertson, ApJS, 2015)

Getting started
----------------
This is the stable branch of the *Cholla* code. *Cholla* is designed to 
be run using NVIDIA GPUs, and can be run in serial mode using one GPU
or with MPI.

Two example makefiles are included in this directory, one designed for
linux and one for mac. After downloading the code, you should
be able to configure it for your machine by modifying one of the makefiles appropriately.


Configuration Notes
------------
Most of the configuration options available in *Cholla* are selected by commenting/uncommenting
the appropriate line within the makefile, e.g. single vs
double precision, output format, the reconstruction method, Riemann solver, integrator, 
and cooling. The entire code must be recompiled any time you change the configuration.

A few options must be specified on the 'FLAGS' line in the makefile. These include
the h correction (-DH_CORRECTION), dual energy (-DDE), the static gravity module (-DSTATIC_GRAV), 
and the passive scalar flag (-DSCALAR). It is strongly recommended that you include the dual energy
flag when cooling is turned on.


Running Cholla
--------------
To run the code after it is compiled, you must supply an input file with parameters and a problem that matches a function
in the initial_conditions file. For example, to run a 1D Sod Shock Tube test, you would do:

./cholla tests/1D/Sod.txt

Some output will be generated in the terminal, and output files will be written in the directory specified
in the input parameter file.

To run *Cholla* in parallel mode, the CHOLLA_MPI flag in the makefile must be uncommented. Then you can run
using:

mpirun -np 4 ./cholla tests/1D/Sod.txt

Each process will be assigned a GPU. *Cholla* cannot be run with more processes than available GPUs,
so MPI mode is most useful on a cluster (or for testing parallel behavior with a single process).


Other Notes
--------------

In practice, we have found the Van Leer integrator to be the most stable. *Cholla* is set to run with a default CFL coefficient of 0.3, but this can be changed within the grid initialization function.

*Cholla* can be run without GPUs by commenting out CUDA in the makefile, but this configuration is not recommended. Because *Cholla*
was designed with GPUs in mind, the CPU performance is, lackluster at best. In addition, some 
of the configuration options are not available in the non-CUDA mode (and warnings are not always included).

When running tests in fewer than 3 dimensions, *Cholla* assumes that the x-direction will be used first, then
the y, then z. This is to say, in 1D nx must always be greater than 1, and in 2D nx and ny must be greater than 1.

Currently, *Cholla* is not designed to be able to run very large (>1e6) 1 dimensional problems. If this is a functionality you are
interested in, please let us know.
