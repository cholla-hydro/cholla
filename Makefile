
DIRS := src src/gravity src/particles src/cosmology src/cooling

CFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

OBJS := $(subst .c,.o,$(CFILES)) $(subst .cpp,.o,$(CPPFILES)) $(subst .cu,.o,$(GPUFILES))
CUOBJS := $(subst .cu,.o,$(GPUFILES))

#To use GPUs, CUDA must be turned on here
#Optional error checking can also be enabled
DFLAGS += -DCUDA #-DCUDA_ERROR_CHECK
# Architecture must be set correctly 
CHOLLA_ARCH ?= sm_70

#To use MPI, DFLAGS must also include -DMPI_CHOLLA
#DFLAGS += -DMPI_CHOLLA -DBLOCK

#Set the MPI Processes grid [nproc_x, nproc_y, nproc_z]
#DFLAGS += -DSET_MPI_GRID

#Limit the number of steps
#DFLAGS += -DN_STEPS_LIMIT=26

# Single or double precision
#DFLAGS += -DPRECISION=1
DFLAGS += -DPRECISION=2

#Set output preferences
DFLAGS += -DOUTPUT
#DFLAGS += -DBINARY
#DFLAGS += -DHDF5
#DFLAGS += -DSLICES
#DFLAGS += -DPROJECTION
#DFLAGS += -DROTATED_PROJECTION

#Output all data every N_OUTPUT_COMPLETE snapshots ( These are Restart Files )
#DFLAGS += -DN_OUTPUT_COMPLETE=10

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
#DFLAGS += -DCTU
DFLAGS += -DVL

# Use Dual Energy Formalism
#DFLAGS += -DDE

# Evolve additional scalars
#DFLAGS += -DSCALAR

# Apply a minimum value to Conserved values
#DFLAGS += -DDENSITY_FLOOR
#DFLAGS += -DTEMPERATURE_FLOOR

# Average Slow cell when the cell delta_t is very small
#DFLAGS += -DAVERAGE_SLOW_CELLS

# Allocate GPU memory every timestep
#DFLAGS += -DDYNAMIC_GPU_ALLOC

# Set the cooling function
#DFLAGS += -DCOOLING_GPU 
#DFLAGS += -DCLOUDY_COOL

# Use Tiled Iitial Conditions for Scaling Tets
#DFLAGS += -DTILED_INITIAL_CONDITIONS

# Print Initial Statistics
#DFLAGS += -DPRINT_INITIAL_STATS

# Print some timing stats
#DFLAGS += -DCPU_TIME

# Include FFT gravity
#DFLAGS += -DGRAVITY
#DFLAGS += -DPFFT
#DFLAGS += -DCUFFT
#DFLAGS += -DCOUPLE_GRAVITATIONAL_WORK
#DFLAGS += -DCOUPLE_DELTA_E_KINETIC
#DFLAGS += -DOUTPUT_POTENTIAL
#DFLAGS += -DGRAVITY_5_POINTS_GRADIENT

# Include Gravity From Particles PM
#DFLAGS += -DPARTICLES
#DFLAGS += -DPARTICLES_CPU
#DFLAGS += -DPARTICLES_GPU
#DFLAGS += -DONLY_PARTICLES
#DFLAGS += -DSINGLE_PARTICLE_MASS
#DFLAGS += -DPARTICLE_IDS

# Turn OpenMP on for CPU calculations
#DFLAGS += -DPARALLEL_OMP
#OMP_NUM_THREADS ?= 16
#DFLAGS += -DN_OMP_THREADS=$(OMP_NUM_THREADS)
#DFLAGS += -DPRINT_OMP_DOMAIN

# Cosmological simulation
#DFLAGS += -DCOSMOLOGY
 
# Use Grackle for cooling in cosmological simulations
#DFLAGS += -DCOOLING_GRACKLE

CC ?= cc
CXX ?= CC
CFLAGS += -g -Ofast
CXXFLAGS += -g -Ofast -std=c++14
CFLAGS += $(DFLAGS) -Isrc
CXXFLAGS += $(DFLAGS) -Isrc
GPUFLAGS += $(DFLAGS) -Isrc

ifeq ($(findstring -DPFFT,$(DFLAGS)),-DPFFT)
	CXXFLAGS += -I$(FFTW_ROOT)/include -I$(PFFT_ROOT)/include
	GPUFLAGS += -I$(FFTW_ROOT)/include -I$(PFFT_ROOT)/include
	LIBS += -L$(FFTW_ROOT)/lib -L$(PFFT_ROOT)/lib -lpfft -lfftw3_mpi -lfftw3
endif

ifeq ($(findstring -DCUFFT,$(DFLAGS)),-DCUFFT)
	LIBS += -lcufft
endif

ifeq ($(findstring -DHDF5,$(DFLAGS)),-DHDF5)
ifneq ($(HDF5_ROOT),)
	CFLAGS += -I$(HDF5_ROOT)/include
	CXXFLAGS += -I$(HDF5_ROOT)/include
	GPUFLAGS += -I$(HDF5_ROOT)/include
	LIBS += -L$(HDF5_ROOT)/lib
endif
	LIBS += -lhdf5
endif

ifeq ($(findstring -DMPI_CHOLLA,$(DFLAGS)),-DMPI_CHOLLA)
	CC = mpicc
	CXX = mpicxx
ifneq ($(MPI_HOME),)
	GPUFLAGS += -I$(MPI_HOME)/include
endif
endif

ifeq ($(findstring -DCUDA,$(DFLAGS)),-DCUDA)
	GPUCXX := nvcc
	GPUFLAGS += --expt-extended-lambda -g -O3 -arch $(CHOLLA_ARCH) -fmad=false
	LD := $(CXX)
	LDFLAGS := $(CXXFLAGS)
ifneq ($(CUDA_DIR),)
	LIBS += -L$(CUDA_DIR)/lib64
endif
	LIBS += -lcudart
endif

ifeq ($(findstring -DCOOLING_GRACKLE,$(DFLAGS)),-DCOOLING_GRACKLE)
	DFLAGS += -DCONFIG_BFLOAT_8
	DFLAGS += -DOUTPUT_TEMPERATURE
	DFLAGS += -DOUTPUT_CHEMISTRY
	#DFLAGS += -DOUTPUT_ELECTRONS
	#DFLAGS += -DOUTPUT_FULL_IONIZATION
	#DFLAGS += -DOUTPUT_METALS
	DFLAGS += -DSCALAR
	DFLAGS += -DN_OMP_THREADS_GRACKLE=12
	CXXFLAGS += -I/ccs/proj/ast149/code/grackle/include
	LIBS += -L/ccs/proj/ast149/code/grackle/lib -lgrackle
endif

ifeq ($(findstring -DPARALLEL_OMP,$(DFLAGS)),-DPARALLEL_OMP)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

.SUFFIXES: .c .cpp .cu .o

EXEC := cholla$(SUFFIX)

$(EXEC): $(OBJS) src/gpuCode.o
	$(LD) $(LDFLAGS) $(OBJS) src/gpuCode.o -o $(EXEC) $(LIBS)

%.o:	%.c
		$(CC) $(CFLAGS) -c $< -o $@

%.o:	%.cpp
		$(CXX) $(CXXFLAGS) -c $< -o $@

%.o:	%.cu
		$(GPUCXX) $(GPUFLAGS) --device-c -c $< -o $@

src/gpuCode.o:	$(CUOBJS)
		$(GPUCXX) -dlink $(GPUFLAGS) $(CUOBJS) -o src/gpuCode.o



.PHONY : clean

clean:
	 rm -f $(OBJS) src/gpuCode.o $(EXEC)


