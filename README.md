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
in the initial_conditions file. E.g. to run a one-dimension Sod Shock Tube test, you would do:

./cholla tests/1D/Sod.txt

Some output will be generated in the terminal, and output files will be written in the directory specified
in the input parameter file.

To run Cholla in parallel mode, you must un-comment the CHOLLA_MPI flag in the makefile. Then you can run
using:

mpirun -np 4 ./cholla tests/1D/Sod.txt

Each process will assign itself to a single GPU. Cholla cannot be run with more processes than available GPUs,
so MPI mode is generally most useful on a cluster, or for testing with a single process.
