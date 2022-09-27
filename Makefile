#-- Set default include makefile
MACHINE ?= $(shell builds/machine.sh)
TYPE    ?= hydro

include builds/make.host.$(MACHINE)
include builds/make.type.$(TYPE)

# CHOLLA_ARCH defaults to sm_70 if not set in make.host
CHOLLA_ARCH ?= sm_70

DIRS     := src src/analysis src/chemistry_gpu src/cooling src/cooling_grackle src/cosmology \
            src/cpu src/global src/gravity src/gravity/paris src/grid src/hydro \
            src/integrators src/io src/main.cpp src/main_tests.cpp \
            src/model src/mpi src/old_cholla src/particles src/reconstruction \
            src/riemann_solvers src/system_tests src/utils src/fft

SUFFIX ?= .$(TYPE).$(MACHINE)

CFILES   := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

# Build a list of all potential object files so cleaning works properly
CLEAN_OBJS := $(subst .c,.o,$(CFILES)) \
              $(subst .cpp,.o,$(CPPFILES)) \
              $(subst .cu,.o,$(GPUFILES))

# Set testing related lists and variables
ifeq ($(TEST), true)
  # This is a test build so lets clear out Cholla's main file and set
  # appropriate compiler flags, suffix, etc
  $(info Building Tests...)
  $(info )
  SUFFIX    := $(strip $(SUFFIX)).tests
  CPPFILES  := $(filter-out src/main.cpp,$(CPPFILES))
  LIBS      += -L$(GOOGLETEST_ROOT)/lib64 -pthread -lgtest -lhdf5_cpp
  TEST_FLAGS = -I$(GOOGLETEST_ROOT)/include
  CFLAGS   = $(TEST_FLAGS)
  CXXFLAGS = $(TEST_FLAGS)
  GPUFLAGS = $(TEST_FLAGS)

  # Set the build flags to debug. This is mostly to avoid the approximations
  # made by Ofast which break std::isnan and std::isinf which are required for
  # the testing
  BUILD = DEBUG
else
  # This isn't a test build so clear out testing related files
  CFILES   := $(filter-out src/system_tests/% %_tests.c,$(CFILES))
  CPPFILES := $(filter-out src/system_tests/% %_tests.cpp,$(CPPFILES))
  CPPFILES := $(filter-out src/utils/testing_utilities.cpp,$(CPPFILES))
  GPUFILES := $(filter-out src/system_tests/% %_tests.cu,$(GPUFILES))
endif

OBJS     := $(subst .c,.o,$(CFILES)) \
            $(subst .cpp,.o,$(CPPFILES)) \
            $(subst .cu,.o,$(GPUFILES))

#-- Set default compilers and flags
CC                ?= cc
CXX               ?= CC

CFLAGS_OPTIMIZE   ?= -Ofast
CXXFLAGS_OPTIMIZE ?= -Ofast -std=c++17
GPUFLAGS_OPTIMIZE ?= -g -O3 -std=c++17
BUILD             ?= OPTIMIZE

CFLAGS            += $(CFLAGS_$(BUILD))
CXXFLAGS          += $(CXXFLAGS_$(BUILD))
GPUFLAGS          += $(GPUFLAGS_$(BUILD))

#-- Add flags and libraries as needed

CFLAGS   += $(DFLAGS) -Isrc
CXXFLAGS += $(DFLAGS) -Isrc
GPUFLAGS += $(DFLAGS) -Isrc

ifeq ($(findstring -DPARIS,$(DFLAGS)),-DPARIS)
  ifdef HIPCONFIG
    CXXFLAGS += -I$(ROCM_PATH)/hipfft/include
    GPUFLAGS += -I$(ROCM_PATH)/hipfft/include
    LIBS += -L$(ROCM_PATH)/hipfft/lib -lhipfft
  else
    LIBS += -lcufft
  endif
  ifeq ($(findstring -DGRAVITY_5_POINTS_GRADIENT,$(DFLAGS)),-DGRAVITY_5_POINTS_GRADIENT)
    DFLAGS += -DPARIS_5PT
  else
    DFLAGS += -DPARIS_3PT
  endif
endif

ifeq ($(findstring -DHDF5,$(DFLAGS)),-DHDF5)
  CXXFLAGS += -I$(HDF5_ROOT)/include
  GPUFLAGS += -I$(HDF5_ROOT)/include
  LIBS     += -L$(HDF5_ROOT)/lib -lhdf5
endif

ifeq ($(findstring -DMPI_CHOLLA,$(DFLAGS)),-DMPI_CHOLLA)
  GPUFLAGS += -I$(MPI_ROOT)/include
  ifdef HIPCONFIG
     LIBS += -L$(MPI_ROOT)/lib -lmpi
  endif
endif

ifeq ($(findstring -DPARALLEL_OMP,$(DFLAGS)),-DPARALLEL_OMP)
  CXXFLAGS += -fopenmp
endif

ifeq ($(findstring -DLYA_STATISTICS,$(DFLAGS)),-DLYA_STATISTICS)
  CXXFLAGS += -I$(FFTW_ROOT)/include
  GPUFLAGS += -I$(FFTW_ROOT)/include
  LIBS += -L$(FFTW_ROOT)/lib -lfftw3_mpi -lfftw3
endif


ifdef HIPCONFIG
  DFLAGS    += -DO_HIP
  CXXFLAGS  += $(HIPCONFIG)
  GPUCXX    ?= hipcc
  GPUFLAGS  += -std=c++17 -Wall -ferror-limit=1
  LD        := $(CXX)
  LDFLAGS   := $(CXXFLAGS)
  LIBS      += -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64
else
  CUDA_INC  ?= -I$(CUDA_ROOT)/include
  CUDA_LIB  ?= -L$(CUDA_ROOT)/lib64 -lcudart
  CXXFLAGS  += $(CUDA_INC)
  GPUCXX    ?= nvcc
  GPUFLAGS  += --expt-extended-lambda -arch $(CHOLLA_ARCH) -fmad=false
  GPUFLAGS  += $(CUDA_INC)
  LD        := $(CXX)
  LDFLAGS   += $(CXXFLAGS)
  LIBS      += $(CUDA_LIB)
endif

ifeq ($(findstring -DCOOLING_GRACKLE,$(DFLAGS)),-DCOOLING_GRACKLE)
  DFLAGS += -DCONFIG_BFLOAT_8
  DFLAGS += -DSCALAR
  CXXFLAGS += -I$(GRACKLE_ROOT)/include
  GPUFLAGS += -I$(GRACKLE_ROOT)/include
  LIBS     += -L$(GRACKLE_ROOT)/lib -lgrackle
endif

ifeq ($(findstring -DCHEMISTRY_GPU,$(DFLAGS)),-DCHEMISTRY_GPU)
  DFLAGS += -DSCALAR
endif

.SUFFIXES: .c .cpp .cu .o

EXEC := bin/cholla$(SUFFIX)

# Get the git hash and setup macro to store a string of all the other macros so
# that they can be written to the save files
DFLAGS      += -DGIT_HASH='"$(shell git rev-parse --verify HEAD)"'
MACRO_FLAGS := -DMACRO_FLAGS='"$(DFLAGS)"'
DFLAGS      += $(MACRO_FLAGS)

$(EXEC): prereq-build $(OBJS)
	mkdir -p bin/ && $(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
	eval $(EXTRA_COMMANDS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f $(CLEAN_OBJS)
	rm -rf googletest
	-find bin/ -type f -executable -name "cholla.*.$(MACHINE)*" -exec rm -f '{}' \;

clobber: clean
	find . -type f -executable -name "cholla*" -exec rm -f '{}' \;
	-find bin/ -type d -name "t*" -prune -exec rm -rf '{}' \;
	rm -rf bin/cholla.*tests*.xml

prereq-build:
	builds/prereq.sh build $(MACHINE)
prereq-run:
	builds/prereq.sh run $(MACHINE)
