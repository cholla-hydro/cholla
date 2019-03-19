EXEC   = cholla

OPTIMIZE =  -O2

DIR = ./src
CFILES = $(wildcard $(DIR)/*.c)
CPPFILES = $(wildcard $(DIR)/*.cpp)
CUDAFILES = $(wildcard $(DIR)/*.cu)

DIR_GRAV = ./src/gravity
CFILES_GRAV = $(wildcard $(DIR_GRAV)/*.c)
CPPFILES_GRAV = $(wildcard $(DIR_GRAV)/*.cpp)
CUDAFILES_GRAV = $(wildcard $(DIR_GRAV)/*.cu)

DIR_PART = ./src/particles
CFILES_PART = $(wildcard $(DIR_PART)/*.c)
CPPFILES_PART = $(wildcard $(DIR_PART)/*.cpp)
CUDAFILES_PART = $(wildcard $(DIR_PART)/*.cu)

OBJS   = $(subst .c,.o,$(CFILES)) $(subst .cpp,.o,$(CPPFILES)) $(subst .cu,.o,$(CUDAFILES)) $(subst .c,.o,$(CFILES_GRAV)) $(subst .cpp,.o,$(CPPFILES_GRAV)) $(subst .cu,.o,$(CUDAFILES_GRAV)) $(subst .c,.o,$(CFILES_PART)) $(subst .cpp,.o,$(CPPFILES_PART)) $(subst .cu,.o,$(CUDAFILES_PART))
COBJS   = $(subst .c,.o,$(CFILES)) $(subst .c,.o,$(CFILES_GRAV)) $(subst .c,.o,$(CFILES_PART))
CPPOBJS   = $(subst .cpp,.o,$(CPPFILES)) $(subst .cpp,.o,$(CPPFILES_GRAV)) $(subst .cpp,.o,$(CPPFILES_PART))
CUOBJS   = $(subst .cu,.o,$(CUDAFILES)) $(subst .cu,.o,$(CUDAFILES_GRAV)) $(subst .cu,.o,$(CUDAFILES_PART))

#To use GPUs, CUDA must be turned on here
#Optional error checking can also be enabled
CUDA = -DCUDA #-DCUDA_ERROR_CHECK

#To use MPI, MPI_FLAGS must be set to -DMPI_CHOLLA
#otherwise gcc/g++ will be used for serial compilation
MPI_FLAGS =  -DMPI_CHOLLA

ifdef MPI_FLAGS
  CC	= mpicc
  CXX   = mpicxx

  #MPI_FLAGS += -DSLAB
  MPI_FLAGS += -DBLOCK

else
  CC	= gcc
  CXX   = g++
endif

#define the NVIDIA CUDA compiler
NVCC	= nvcc

.SUFFIXES : .c .cpp .cu .o

#PRECISION = -DPRECISION=1
PRECISION = -DPRECISION=2

#OUTPUT = -DBINARY
OUTPUT = -DHDF5

#RECONSTRUCTION = -DPCM
#RECONSTRUCTION = -DPLMP
# RECONSTRUCTION = -DPLMC
#RECONSTRUCTION = -DPPMP
RECONSTRUCTION = -DPPMC

#SOLVER = -DEXACT
#SOLVER = -DROE
SOLVER = -DHLLC

#INTEGRATOR = -DCTU
INTEGRATOR = -DVL

#Dual Energy Formalism
DUAL_ENERGY = -DDE

#Allocate GPU memory only once at the first timestep
SINGLE_ALLOC_GPU = -DSINGLE_ALLOC_GPU

COOLING = #-DCOOLING_GPU -DCLOUDY_COOL

CPU_TIME = -DCPU_TIME

#INCLUDE GRAVITY
GRAVITY = -DGRAVITY
POISSON_SOLVER = -DPFFT
# GRAVITY_COUPLE = -DGRAVITY_COUPLE_GPU
GRAVITY_COUPLE = -DGRAVITY_COUPLE_CPU
# GRAVITY_ENERGY_COUPLE = -DCOUPLE_GRAVITATIONAL_WORK
GRAVITY_ENERGY_COUPLE = -DCOUPLE_DELTA_E_KINETIC
OUTPUT_POTENTIAL = -DOUTPUT_POTENTIAL

#Include Gravity From Particles PM
PARTICLES = -DPARTICLES
PARTICLES_INT = -DPARTICLES_LONG_INTS
PARTICLE_IDS = -DPARTICLE_IDS
ONLY_PARTICLES = -DONLY_PARTICLES

#TURN OMP ON FOR CPU CALCULATIONS
PARALLEL_OMP = -DPARALLEL_OMP
N_OMP_THREADS = -DN_OMP_THREADS=10
# PRINT_OMP_DOMAIN = -DPRINT_OMP_DOMAIN


ifdef CUDA
CUDA_INCL = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart
endif
ifeq ($(OUTPUT),-DHDF5)
HDF5_INCL = -I/usr/include/hdf5/serial/
HDF5_LIBS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
endif

INCL   = -I./ $(HDF5_INCL)
NVINCL = $(INCL) $(CUDA_INCL)
LIBS   = -lm $(HDF5_LIBS) $(CUDA_LIBS)

ifeq ($(POISSON_SOLVER),-DPFFT)
FFTW_INCL = -I/home/bruno/apps/fftw-3.3.5/include
FFTW_LIBS = -L/home/bruno/apps/fftw-3.3.5/lib -lfftw3
PFFT_INCL = -I/home/bruno/apps/pfft-1.0.8-alpha/include
PFFT_LIBS = -L/home/bruno/apps/pfft-1.0.8-alpha/lib  -lpfft  -lfftw3_mpi -lfftw3
INCL += $(FFTW_INCL) $(PFFT_INCL)
LIBS += $(FFTW_LIBS) $(PFFT_LIBS)
endif

ifdef PARALLEL_OMP
OMP_FLAGS = -fopenmp
LIBS += -fopenmp
endif


FLAGS_HYDRO = $(CUDA) $(PRECISION) $(OUTPUT) $(RECONSTRUCTION) $(SOLVER) $(INTEGRATOR) $(DUAL_ENERGY) $(COOLING) $(SINGLE_ALLOC_GPU) $(CPU_TIME)#-DSTATIC_GRAV #-DDE -DSCALAR -DSLICES -DPROJECTION -DROTATED_PROJECTION
FLAGS_OMP = $(PARALLEL_OMP) $(N_OMP_THREADS) $(PRINT_OMP_DOMAIN)
FLAGS_GRAVITY = $(GRAVITY) $(POISSON_SOLVER) $(GRAVITY_COUPLE) $(GRAVITY_ENERGY_COUPLE) $(OUTPUT_POTENTIAL)
FLAGS_PARTICLES = $(PARTICLES) $(PARTICLES_INT) $(PARTICLE_IDS) $(ONLY_PARTICLES)
FLAGS = $(FLAGS_HYDRO) $(FLAGS_OMP) $(FLAGS_GRAVITY) $(FLAGS_PARTICLES) 
CFLAGS 	  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) $(OMP_FLAGS)
CXXFLAGS  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) $(OMP_FLAGS)
NVCCFLAGS = $(FLAGS) -fmad=false -ccbin=$(CC)


%.o:	%.c
		$(CC) $(CFLAGS)  $(INCL)  -c $< -o $@

%.o:	%.cpp
		$(CXX) $(CXXFLAGS)  $(INCL) -c $< -o $@

%.o:	%.cu
		$(NVCC) $(NVCCFLAGS) --device-c $(NVINCL)  -c $< -o $@

$(EXEC): $(OBJS) src/gpuCode.o
	 	 $(CXX) $(OBJS) src/gpuCode.o $(LIBS) -o $(EXEC)

src/gpuCode.o:	$(CUOBJS)
		$(NVCC) -dlink $(CUOBJS) -o src/gpuCode.o



.PHONY : clean

clean:
	 rm -f $(OBJS) src/gpuCode.o $(EXEC)
