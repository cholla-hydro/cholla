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

DIR_COSMO = ./src/cosmology
CFILES_COSMO = $(wildcard $(DIR_COSMO)/*.c)
CPPFILES_COSMO = $(wildcard $(DIR_COSMO)/*.cpp)
CUDAFILES_COSMO = $(wildcard $(DIR_COSMO)/*.cu)

DIR_COOL = ./src/cooling
CFILES_COOL = $(wildcard $(DIR_COOL)/*.c)
CPPFILES_COOL = $(wildcard $(DIR_COOL)/*.cpp)
CUDAFILES_COOL = $(wildcard $(DIR_COOL)/*.cu)


OBJS   = $(subst .c,.o,$(CFILES)) $(subst .cpp,.o,$(CPPFILES)) $(subst .cu,.o,$(CUDAFILES)) $(subst .c,.o,$(CFILES_GRAV)) $(subst .cpp,.o,$(CPPFILES_GRAV)) $(subst .cu,.o,$(CUDAFILES_GRAV)) $(subst .c,.o,$(CFILES_PART)) $(subst .cpp,.o,$(CPPFILES_PART)) $(subst .cu,.o,$(CUDAFILES_PART)) $(subst .c,.o,$(CFILES_COSMO)) $(subst .cpp,.o,$(CPPFILES_COSMO)) $(subst .cu,.o,$(CUDAFILES_COSMO)) $(subst .c,.o,$(CFILES_COOL)) $(subst .cpp,.o,$(CPPFILES_COOL)) $(subst .cu,.o,$(CUDAFILES_COOL))
COBJS   = $(subst .c,.o,$(CFILES)) $(subst .c,.o,$(CFILES_GRAV)) $(subst .c,.o,$(CFILES_PART)) $(subst .c,.o,$(CFILES_COSMO))  $(subst .c,.o,$(CFILES_COOL))
CPPOBJS   = $(subst .cpp,.o,$(CPPFILES)) $(subst .cpp,.o,$(CPPFILES_GRAV)) $(subst .cpp,.o,$(CPPFILES_PART)) $(subst .cpp,.o,$(CPPFILES_COSMO)) $(subst .cpp,.o,$(CPPFILES_COOL))
CUOBJS   = $(subst .cu,.o,$(CUDAFILES)) $(subst .cu,.o,$(CUDAFILES_GRAV)) $(subst .cu,.o,$(CUDAFILES_PART)) $(subst .cu,.o,$(CUDAFILES_COSMO)) $(subst .cu,.o,$(CUDAFILES_COOL))

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

#Limit the number of steps
# N_STEPS_LIMIT = -DN_STEPS_LIMIT=1

#PRECISION = -DPRECISION=1
PRECISION = -DPRECISION=2

#OUTPUT = -DBINARY
OUTPUT = -DHDF5

# RECONSTRUCTION = -DPCM
# RECONSTRUCTION = -DPLMP
# RECONSTRUCTION = -DPLMC
RECONSTRUCTION = -DPPMP
# RECONSTRUCTION = -DPPMC

# SOLVER = -DEXACT
# SOLVER = -DROE
SOLVER = -DHLLC
# SOLVER = -DHLL

# INTEGRATOR = -DCTU
# INTEGRATOR = -DVL
INTEGRATOR = -DSIMPLE

#Dual Energy Formalism
DUAL_ENERGY = -DDE
# OUTPUT_DUAL_ENERGY_FLAGS = -DOUTPUT_DUAL_ENERGY_FLAGS

#Fragmentation Condition
# JEANS_CONDITION = -DJEANS_CONDITION

#Apply a minimum value to Conserved values
DENSITY_FLOOR = -DDENSITY_FLOOR
TEMPERATURE_FLOOR = -DTEMPERATURE_FLOOR

#Output every time step
# OUTPUT_ALWAYS = -DOUTPUT_ALWAYS

#Average Slow cells
# AVERAGE_SLOW_CELLS = -DAVERAGE_SLOW_CELLS

#Allocate GPU memory only once at the first timestep
# DYNAMIC_GPU_ALLOC = -DDYNAMIC_GPU_ALLOC

COOLING = #-DCOOLING_GPU -DCLOUDY_COOL

#Use Tiled Iitial Conditions for Scaling Tets
# TILED_INITIAL_CONDITIONS = -DTILED_INITIAL_CONDITIONS


#Print Initial Statistics
PRINT_INITIAL_STATS = -DPRINT_INITIAL_STATS

#Meassure and print times for each component
CPU_TIME = -DCPU_TIME

#INCLUDE GRAVITY
GRAVITY = -DGRAVITY
POISSON_SOLVER = -DPFFT
# POISSON_SOLVER = -DCUFFT
GRAVITY_INT = -DGRAVITY_LONG_INTS
# GRAVITY_COUPLE = -DGRAVITY_COUPLE_GPU
GRAVITY_COUPLE = -DGRAVITY_COUPLE_CPU
GRAVITY_ENERGY_COUPLE = -DCOUPLE_GRAVITATIONAL_WORK
# GRAVITY_ENERGY_COUPLE = -DCOUPLE_DELTA_E_KINETIC
# OUTPUT_POTENTIAL = -DOUTPUT_POTENTIAL
GRAVITY_GRADIENT = -DGRAVITY_5_POINTS_GRADIENT

#Include Gravity From Particles PM
PARTICLES = -DPARTICLES
# ONLY_PARTICLES = -DONLY_PARTICLES
SINGLE_PARTICLE_MASS = -DSINGLE_PARTICLE_MASS
PARTICLES_INT = -DPARTICLES_LONG_INTS
PARTICLES_INTEGRATOR = -DPARTICLES_KDK
# PARTICLES_INTEGRATOR = -DPARTICLES_DKD
PARTICLES_PECULIAR_VELOCITIES = -DPARTICLES_PECULIAR_VELOCITIES
# PARTICLE_IDS = -DPARTICLE_IDS

# TURN OMP ON FOR CPU CALCULATIONS
PARALLEL_OMP = -DPARALLEL_OMP
N_OMP_THREADS = -DN_OMP_THREADS=10
# # PRINT_OMP_DOMAIN = -DPRINT_OMP_DOMAIN

#Cosmological simulation
COSMOLOGY = -DCOSMOLOGY

#Use Grackle for cooling in cosmological simulations
COOLING = -DCOOLING_GRACKLE

ifdef CUDA
CUDA_INCL = -I/cm/shared/apps/cuda10.1/toolkit/current/include
CUDA_LIBS = -L/cm/shared/apps/cuda10.1/toolkit/current/targets/x86_64-linux/lib/stubs/ -lcuda -lcudart
endif
ifeq ($(OUTPUT),-DHDF5)
HDF5_INCL = -I/cm/shared/apps/hdf5/1.10.1/include
HDF5_LIBS = -L/cm/shared/apps/hdf5/1.10.1/lib -lhdf5
endif

INCL   = -I./ $(HDF5_INCL)
NVINCL = $(INCL) $(CUDA_INCL)
LIBS   = -lm $(HDF5_LIBS) $(CUDA_LIBS)

ifeq ($(POISSON_SOLVER),-DPFFT)
FFTW_INCL = -I/cm/shared/apps/fftw/fftw-3.3.8/include
FFTW_LIBS = -L/cm/shared/apps/fftw/fftw-3.3.8/lib -lfftw3
PFFT_INCL = -I/home/brvillas/code/pfft/include
PFFT_LIBS = -L/home/brvillas/code/pfft/lib  -lpfft  -lfftw3_mpi -lfftw3
INCL += $(FFTW_INCL) $(PFFT_INCL)
LIBS += $(FFTW_LIBS) $(PFFT_LIBS)
endif

ifeq ($(POISSON_SOLVER),-DCUFFT)
CUFFT_INCL = -I/usr/local/cuda-9.0/targets/x86_64-linux/include
CUFFT_LIBS = -L/usr/local/cuda-9.0/targets/x86_64-linux/lib -lcufft
INCL += $(CUFFT_INCL) 
LIBS += $(CUFFT_LIBS) 
endif

ifeq ($(COOLING),-DCOOLING_GRACKLE)
GRACKLE_PRECISION = -DCONFIG_BFLOAT_8
OUTPUT_TEMPERATURE = -DOUTPUT_TEMPERATURE
OUTPUT_CHEMISTRY = -DOUTPUT_CHEMISTRY
SCALAR = -DSCALAR
N_OMP_THREADS_GRACKLE = -DN_OMP_THREADS_GRACKLE=10
GRACKLE_INCL = -I/home/brvillas/code/grackle/include
GRACKLE_LIBS = -L/home/brvillas/code/grackle/lib -lgrackle 
INCL += $(GRACKLE_INCL)
LIBS += $(GRACKLE_LIBS)
endif

ifdef PARALLEL_OMP
OMP_FLAGS = -fopenmp
LIBS += -fopenmp
endif


FLAGS_HYDRO = $(CUDA) $(PRECISION) $(OUTPUT) $(RECONSTRUCTION) $(SOLVER) $(INTEGRATOR) $(DUAL_ENERGY) $(COOLING) $(DYNAMIC_GPU_ALLOC) $(CPU_TIME) $(PRINT_INITIAL_STATS) $(DENSITY_FLOOR) $(TEMPERATURE_FLOOR) $(SCALAR) $(AVERAGE_SLOW_CELLS) $(OUTPUT_ALWAYS) $(TILED_INITIAL_CONDITIONS) $(N_STEPS_LIMIT) $(JEANS_CONDITION) $(OUTPUT_DUAL_ENERGY_FLAGS)
FLAGS_OMP = $(PARALLEL_OMP) $(N_OMP_THREADS) $(PRINT_OMP_DOMAIN)
FLAGS_GRAVITY = $(GRAVITY) $(POISSON_SOLVER) $(GRAVITY_COUPLE) $(GRAVITY_ENERGY_COUPLE) $(OUTPUT_POTENTIAL) $(GRAVITY_INT) $(GRAVITY_GRADIENT)
FLAGS_PARTICLES = $(PARTICLES) $(PARTICLES_INTEGRATOR) $(PARTICLES_PECULIAR_VELOCITIES) $(PARTICLES_INT) $(PARTICLE_IDS) $(ONLY_PARTICLES) $(SINGLE_PARTICLE_MASS)
FLAGS_COSMO = $(COSMOLOGY)
FLAGS_COOLING = $(COOLING) $(GRACKLE_PRECISION) $(OUTPUT_TEMPERATURE) $(OUTPUT_CHEMISTRY) $(N_OMP_THREADS_GRACKLE)
FLAGS = $(FLAGS_HYDRO) $(FLAGS_OMP) $(FLAGS_GRAVITY) $(FLAGS_PARTICLES) $(FLAGS_COSMO) $(FLAGS_COOLING)
CFLAGS 	  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) $(OMP_FLAGS)
CXXFLAGS  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) $(OMP_FLAGS)
NVCCFLAGS = $(FLAGS) --fmad=false -ccbin=$(CC)


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
