Getting started
===============


Requirements
------------
- An NVIDIA graphics card
- A C/C++ compiler, such as g++
- The NVIDIA cuda compiler, nvcc (the CUDA toolkit is available `here <https://developer.nvidia.com/accelerated-computing-toolkit>`_.
- The hdf5 library (recommended)

Downloading the code
--------------------
The public version of Cholla can be found at https://github.com/cholla-hydro/cholla. To download it, you can either clone the main repository directly

``git clone https://github.com/cholla-hydro/cholla``

or create your own fork on github and clone that (recommended if you plan to contribute).

Compiling
---------

The main repository contains a Makefile that is used to configure the code. Once you have downloaded the required compilers and libraries, you should be able to compile Cholla by typing ``make`` in the top-level directory. If successful, this will create an executable called "cholla" in the same directory. You will probably have to edit the makefile to tell it where the libraries are. If you are running Cholla on a cluster, you may also need to load the relevant modules prior to compiling the code.

Note: It is important that the code be compiled for the correct GPU architecture. This is specified in the makefile via the -arch flag. The GPU architecture can be found by running the "Device_Query" sample program from the NVIDIA Cuda toolkit (located in the "Samples/Utilities" folder wherever Cuda was installed). Several common architectures are -arch=sm_35 for Tesla K20's, -arch=sm_60 for Tesla P100's, or -arch=sm_70 for Tesla V100's.

Running (serial mode)
---------------------

To run cholla on a single GPU, you execute the binary and provide it with an input parameter file. For example, to run a 1D Sod Shock tube test, within the top-level directory you would type:

``./cholla examples/1D/Sod.txt``

The code will write some information about the input parameters to the terminal:

::

  Parameter values:  nx = 100, ny = 1, nz = 1, tout = 0.200000, init = Riemann, boundaries = 3 3 0 0 0 0
  Output directory:  ./
  Local number of grid cells: 100 1 1 106

followed by some text indicating that the code is initializing:

::

  Setting initial conditions...
  Initial conditions set.
  Setting boundary conditions...
  Boundary conditions set.
  Dimensions of each cell: dx = 0.010000 dy = 0.010000 dz = 0.010000
  Ratio of specific heats gamma = 1.400000
  Nstep = 0  Timestep = 0.000000  Simulation time = 0.000000
  Writing initial conditions to file...
  Starting calculations.

After this, the code will print out a line for every time step it takes, indicating the step it is on, the total time elapsed in the simulation, the size of the timestep taken, the wall-time elapsed during the timestep, and the total wall-time of the simulation:

::

  n_step: 1   sim time:  0.0025355   sim timestep: 2.5355e-03  timestep time =   678.762 ms   total time =    0.6972 s

The code will stop running when it reaches the final time specified in the input file. If the OUTPUT flag was turned on, it will also have created at least 2 output files in the output directory specified in the parameter file (in this case, the same directory where we ran the code), one for the initial conditions and one for the final output. Additional files may have been created depending on the timestep chosen for outputs in the parameter file.

Running (parallel mode)
-----------------------

Cholla can also be run using multiple GPUs when it is compiled using the Message Passing Interface (MPI) protocol. To run in parallel mode requires an mpi compiler, such as openmpi. Once the mpi compiler is installed or loaded, uncomment the relevant line in the makefile:

``MPI_FLAGS =  -DMPI_CHOLLA``

and compile the code. (If you have already compiled the code in serial mode, be sure to clean up first: ``make clean``.) Once the code is compiled with mpi, you can run it using as many processes as you have available GPUs - Cholla assumes there is one GPU per MPI process. For example, if you have 4 GPUs, you could run a 3D sound wave test via:

``mpirun -np 4 ./cholla examples/3D/sound_wave.txt``

The code will automatically divide the simulation domain amongst the GPUs. If you are running on a cluster, you may have to specify additional information about the number of GPUs per node in the batch submission script (e.g. PBS, slurm, LSF).
