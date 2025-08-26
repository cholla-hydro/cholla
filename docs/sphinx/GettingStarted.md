# Getting Started

## Requirements
* An NVIDIA or AMD graphics card
* A C++ compiler, such as g++ (it hasn't been tested recently with other compilers)
* The NVIDIA CUDA compiler, nvcc (the CUDA toolkit is available [here](https://developer.nvidia.com/accelerated-computing-toolkit)), or the AMD HIP compiler (see the [hip installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/HIP-Installation.html))
* The hdf5 library (recommended)

## Downloading the code
The public version of Cholla can be found at [https://github.com/cholla-hydro/cholla](https://github.com/cholla-hydro/cholla).
To download it, you can either clone the main repository directly

```shell-session
$ git clone https://github.com/cholla-hydro/cholla
```

or create your own fork on github and clone that (recommended if you plan to contribute).

## Compiling the code

The main repository contains a Makefile that is used to configure the code, and a 'builds' directory where you can define your own make types (see [Compiling Cholla](./CompilingCholla.md)).
Once you have downloaded the required compilers and libraries and added your machine to the builds, you should be able to compile Cholla by typing

```shell-session
$ make
```

in the top-level directory. If successful, this will create an executable called `"cholla.[host].[type]"` in the 'bin' directory, where `[host]` and `[type]` correspond to your machine name and the build type you selected (hydro, gravity, particles, etc.).

Note: It is important that the code be compiled for the correct GPU architecture.
For NVIDIA GPUs, this is specified in the make.host file via the -arch flag.
The GPU architecture can be found by running the "Device\_Query" sample program from the NVIDIA Cuda toolkit (located in the "Samples/Utilities" folder wherever Cuda was installed).
Several common architectures are -arch=sm\_35 for Tesla K20's, -arch=sm\_60 for Tesla P100's, or -arch=sm\_70 for Tesla V100's.
For AMD GPUs, it is specified using the --offload-arch flag (see make.host.frontier for an example for MI250X).

## Running the code (serial mode)

To run cholla on a single GPU, you execute the binary and provide it with an input parameter file.
For example, to run a 1D Sod Shock tube test, within the top-level directory you would type:

```shell-session
$ .bin/cholla.[yourhost].hydro examples/1D/Sod.txt
```

The code will write some information about the input parameters to the terminal:

```output
Parameter values:  nx = 100, ny = 1, nz = 1, tout = 0.200000, init = Riemann, boundaries = 3 3 0 0 0 0
Output directory:  ./
Local number of grid cells: 100 1 1 106
```

followed by some text indicating that the code is initializing:

```output
Setting initial conditions...
Initial conditions set.
Setting boundary conditions...
Boundary conditions set.
Dimensions of each cell: dx = 0.010000 dy = 0.010000 dz = 0.010000
Ratio of specific heats gamma = 1.400000
Nstep = 0  Timestep = 0.000000  Simulation time = 0.000000
Writing initial conditions to file...
Starting calculations.
```

After this, the code will print out a line for every time step it takes, indicating the step it is on, the total time elapsed in the simulation, the size of the timestep taken, the wall-time elapsed during the timestep, and the total wall-time of the simulation:

```output
n_step: 1   sim time:  0.0025355   sim timestep: 2.5355e-03  timestep time =   678.762 ms   total time =    0.6972 s
```

The code will stop running when it reaches the final time specified in the input file. If the OUTPUT flag was turned on, it will also have created at least 2 output files in the output directory specified in the parameter file (in this case, the same directory where we ran the code), one for the initial conditions and one for the final output. Additional files may have been created depending on the timestep chosen for outputs in the parameter file.

## Running the code (parallel mode)

Cholla can also be run using multiple GPUs when it is compiled using the Message Passing Interface (MPI) protocol. To run in parallel mode requires an mpi compiler, such as openmpi. Once the mpi compiler is installed or loaded, uncomment the relevant line in the makefile:

```make
MPI_FLAGS =  -DMPI_CHOLLA
```

and compile the code. (If you have already compiled the code in serial mode, be sure to clean up first: `make clean`.) Once the code is compiled with mpi, you can run it using as many processes as you have available GPUs - Cholla assumes there is one GPU per MPI process. For example, if you have 4 GPUs, you could run a 3D sound wave test via:

```shell-session
$ mpirun -np 4 ./bin/cholla.[yourhost].hydro examples/3D/sound_wave.txt
```

The code will automatically divide the simulation domain amongst the GPUs. If you are running on a cluster, you may have to specify additional information about the number of GPUs per node in the batch submission script (e.g. PBS, slurm, LSF).
