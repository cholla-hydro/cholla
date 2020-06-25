
DIRS := src src/gravity src/particles src/cosmology src/cooling
ifeq ($(findstring -DPARIS,$(POISSON_SOLVER)),-DPARIS)
  DIRS += src/gravity/paris
endif

CFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

OBJS := $(subst .c,.o,$(CFILES)) $(subst .cpp,.o,$(CPPFILES)) $(subst .cu,.o,$(GPUFILES))


#To use GPUs, CUDA must be turned on here
#Optional error checking can also be enabled
DFLAGS = -DCUDA #-DCUDA_ERROR_CHECK

#To use MPI, MPI_FLAGS must be set to -DMPI_CHOLLA
#otherwise gcc/g++ will be used for serial compilation
DFLAGS =  -DMPI_CHOLLA -DBLOCK

# Single or double precision
#PRECISION = -DPRECISION=1
DFLAGS = -DPRECISION=2

# Output format
#DFLAGS += -DBINARY
DFLAGS += -DHDF5

#Output all data every N_OUTPUT_COMPLETE snapshots ( These are Restart Files )
DFLAGS += -DN_OUTPUT_COMPLETE=1

# Reconstruction
#DFLAGS += -DPCM
#DFLAGS += -DPLMP
#DFLAGS += -DPLMC
DFLAGS += -DPPMP
#DFLAGS += -DPPMC

# Riemann Solver
#DFLAGS += -DEXACT
#DFLAGS += -DROE
DFLAGS += -DHLLC

# Integrator
# DFLAGS += -DVL
DFLAGS += -DSIMPLE

# Dual-Energy Formalism
DFLAGS += -DDE

# Apply a minimum value to conserved values
DFLAGS += -DDENSITY_FLOOR
DFLAGS += -DTEMPERATURE_FLOOR

#Average Slow cell when the cell delta_t is very small
#AVERAGE_SLOW_CELLS = -DAVERAGE_SLOW_CELLS

# Allocate GPU memory every timestep
#DFLAGS += -DDYNAMIC_GPU_ALLOC

# Set the cooling function
#DFLAGS += -DCOOLING_GPU
#DFLAGS += -DCLOUDY_COOL

# Use tiled initial conditions for scaling tests
#DFLAGS += -DTILED_INITIAL_CONDITIONS

#Average Slow cell when the cell delta_t is very small
# DFLAGS += -DAVERAGE_SLOW_CELLS

#Print Initial Statistics
# DFLAGS += -DPRINT_INITIAL_STATS

#Measure Timing of different stages
DFLAGS += -DCPU_TIME

#Gravity Flags
DFLAGS += -DGRAVITY
DFLAGS += -DGRAVITY_LONG_INTS
DFLAGS += -DCOUPLE_GRAVITATIONAL_WORK
#DFLAGS += -DCOUPLE_DELTA_E_KINETIC
DFLAGS += -DOUTPUT_POTENTIAL
DFLAGS += -DGRAVITY_5_POINTS_GRADIENT
POISSON_SOLVER ?= -DPFFT
DFLAGS += $(POISSON_SOLVER)

# Include gravity from particles PM
DFLAGS += -DPARTICLES
DFLAGS += -DPARTICLES_CPU
# DFLAGS += -DONLY_PARTICLES
DFLAGS += -DSINGLE_PARTICLE_MASS
DFLAGS += -DPARTICLES_LONG_INTS
DFLAGS += -DPARTICLES_KDK

# Turn OpenMP on for CPU calculations
DFLAGS += -DPARALLEL_OMP
OMP_NUM_THREADS ?= 16
DFLAGS += -DN_OMP_THREADS=$(OMP_NUM_THREADS)
#DFLAGS += -DPRINT_OMP_DOMAIN

# Cosmology simulation
DFLAGS += -DCOSMOLOGY

# Use Grackle for cooling in cosmological simulations
# DFLAGS += -DCOOLING_GRACKLE -DCONFIG_BFLOAT_8 -DOUTPUT_TEMPERATURE -DOUTPUT_CHEMISTRY -DSCALAR -DN_OMP_THREADS_GRACKLE=20


SYSTEM = "Lux"


ifdef HIP_PLATFORM
  DFLAGS += -DO_HIP
  CXXFLAGS += -D__HIP_PLATFORM_HCC__
  ifeq ($(findstring -DPARIS,$(DFLAGS)),-DPARIS)
    DFLAGS += -I$(ROCM_PATH)/include
  endif
endif



ifeq ($(SYSTEM),"Poplar")
CC := cc
CXX := CC
CXXFLAGS += -std=c++17 -ferror-limit=1
endif


ifeq ($(SYSTEM),"Lux")
CC = gcc
CXX = g++
ifeq ($(findstring -DMPI_CHOLLA,$(DFLAGS)),-DMPI_CHOLLA)
CC = mpicc
CXX = mpic++
endif
CXXFLAGS += -std=c++11
GPUFLAGS += -std=c++11
DFLAGS += -DPARIS_NO_GPU_MPI
OMP_NUM_THREADS = 20
FFTW_ROOT = /data/groups/comp-astro/bruno/code_mpi_local/fftw-3.3.8
PFFT_ROOT = /data/groups/comp-astro/bruno/code_mpi_local/pfft
GRAKLE_HOME = /home/brvillas/code/grackle
endif

CC ?= cc
CXX ?= CC
CFLAGS += -g -Ofast
CXXFLAGS += -g -Ofast 
CFLAGS += $(DFLAGS) -Isrc
CXXFLAGS += $(DFLAGS) -Isrc
GPUFLAGS += $(DFLAGS) -Isrc

ifeq ($(findstring -DPFFT,$(DFLAGS)),-DPFFT)
  CXXFLAGS += -I$(FFTW_ROOT)/include -I$(PFFT_ROOT)/include
  GPUFLAGS += -I$(FFTW_ROOT)/include -I$(PFFT_ROOT)/include
  LIBS += -L$(FFTW_ROOT)/lib -L$(PFFT_ROOT)/lib -lpfft -lfftw3_mpi -lfftw3
endif

ifeq ($(findstring -DCUFFT,$(DFLAGS)),-DCUFFT)
  ifdef HIP_PLATFORM
    LIBS += -L$(ROCM_PATH)/lib -lrocfft
  else
    LIBS += -lcufft
  endif
endif

ifeq ($(findstring -DPARIS,$(DFLAGS)),-DPARIS)
  ifdef HIP_PLATFORM
    LIBS += -L$(ROCM_PATH)/lib -lrocfft
  else
    LIBS += -lcufft
  endif
endif

ifeq ($(findstring -DHDF5,$(DFLAGS)),-DHDF5)
  CXXFLAGS += -I$(HDF5INCLUDE)
  GPUFLAGS += -I$(HDF5INCLUDE)
  LIBS += -L$(HDF5DIR) -lhdf5
endif

ifeq ($(findstring -DMPI_CHOLLA,$(DFLAGS)),-DMPI_CHOLLA)
  GPUFLAGS += -I$(MPI_HOME)/include
	CXXFLAGS += -I$(MPI_HOME)/include
  ifdef HIP_PLATFORM
    LIBS += -L$(MPI_HOME)/lib -lmpi
  endif
endif

ifdef HIP_PLATFORM
  CXXFLAGS += -I$(ROCM_PATH)/include -Wno-unused-result
  GPUCXX := hipcc
  GPUFLAGS += -g -Ofast -Wall --amdgpu-target=gfx906 -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-command-line-argument -Wno-duplicate-decl-specifier -std=c++14 -ferror-limit=1
  LD := $(GPUCXX)
  LDFLAGS += $(GPUFLAGS)
else
  GPUCXX := nvcc
  GPUFLAGS += --expt-extended-lambda -g -O3 -arch sm_70 -fmad=false
  LD := $(CXX)
  LDFLAGS += $(CXXFLAGS)
  LIBS += -lcudart
endif

ifeq ($(findstring -DPARALLEL_OMP,$(DFLAGS)),-DPARALLEL_OMP)
  CXXFLAGS += -fopenmp
  ifdef HIP_PLATFORM
    LIBS += -L$(CRAYLIBS_X86_64) -L$(GCC_X86_64)/lib64 -lcraymath -lcraymp -lu
  else
    LDFLAGS += -fopenmp
  endif
endif


ifeq ($(findstring -DCOOLING_GRACKLE,$(DFLAGS)),-DCOOLING_GRACKLE)
  CXXFLAGS += -I$(GRAKLE_HOME)/include
	CPUFLAGS += -I$(GRAKLE_HOME)/include
	GPUFLAGS += -I$(GRAKLE_HOME)/include
  LIBS += -L$(GRAKLE_HOME)/lib -lgrackle 
endif



.SUFFIXES: .c .cpp .cu .o

EXEC := cholla$(SUFFIX)

$(EXEC): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f $(OBJS) $(EXEC)


