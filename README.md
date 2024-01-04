![Build & Lint](https://github.com/cholla-hydro/cholla/actions/workflows/build_and_lint.yml/badge.svg)
![Code Formatting](https://github.com/cholla-hydro/cholla/actions/workflows/code_formatting.yml/badge.svg)

CHOLLA
============
A 3D GPU-based hydrodynamics code (Schneider & Robertson, ApJS, 2015).


https://user-images.githubusercontent.com/3432028/188235319-e5eb4e5e-00c6-435f-a2f3-7e9f7929c883.mp4


Getting started
----------------
This is the stable branch of the *Cholla* hydrodynamics code.

*Cholla* is designed to be run using (AMD or NVIDIA) GPUs, and can be run in serial mode using one GPU
or with MPI for multiple GPUs.

See the Wiki associated with this repository for more details.


Configuration Notes
------------
Most of the configuration options available in *Cholla* are selected by commenting/uncommenting
the appropriate line in a make.type file or by setting environment variables in a make.host file. A Makefile is included in the top level directory, but this Makefile is not intended to be modified directly. Instead, after downloading the code, you should
be able to configure it for your machine by creating a build script based on one of the make.host examples in the builds directory (see the wiki for more details). Examples of configurations that require edits to a make.type file include single vs
double precision, output format, the reconstruction method, Riemann solver, integrator, and cooling. Examples of configurations that require edits to a make.host file include library paths, compiler options, and gpu-enabled MPI. The entire code must be recompiled any time you change the configuration. For more information on the various options, see the "[Makefile](https://github.com/cholla-hydro/cholla/wiki/Makefile-Parameters)" page of the wiki.


Running Cholla
--------------
To run the code after it is compiled, you must supply an input file with parameters and a problem that matches a function
in the `initial_conditions` file. For example, to run a 1D Sod Shock Tube test, you would type

```./bin/cholla.[host].hydro examples/1D/Sod.txt```

in the directory with the `cholla` binary (where [host] depends on your machine name). Some output will be generated in the terminal, and output files will be written in the directory specified in the input parameter file. More information on input parameters can be found on the "[Input File](https://github.com/cholla-hydro/cholla/wiki/Input-File-Parameters)" page of the wiki.

To run *Cholla* in parallel mode, the `CHOLLA_MPI` macro must be defined at compile time, by including it in the build script (see make.type.hydro for an example). Then you can run
using a command like the following:

```srun -n4 ./cholla.[host].hydro examples/3D/sound_wave.txt```

Each process will be assigned a GPU. *Cholla* cannot be run with more processes than available GPUs,
so MPI mode is most useful on a cluster (or for testing parallel behavior with a single process). Note that more recent AMD devices have 2 GPUs (or GCDs) per accelerator device, so you can run with 2x the number of MPI tasks.

More information about compiling and running *Cholla* can be found in the wiki associated with this repository.

Other Notes
--------------

When running in fewer than 3 dimensions, *Cholla* assumes that the X direction will be used first, then
the Y, then Z. This is to say, in 1D `nx` must always be greater than 1, and in 2D `nx` and `ny` must be greater than 1.
