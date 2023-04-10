SHELL = /usr/bin/env bash
#-- Set default include makefile
MACHINE ?= $(shell builds/machine.sh)
TYPE    ?= hydro

include builds/make.host.$(MACHINE)
include builds/make.type.$(TYPE)

# CUDA_ARCH defaults to sm_70 if not set in make.host
CUDA_ARCH ?= sm_70

DIRS     := src src/analysis src/chemistry_gpu src/cooling src/cooling_grackle src/cosmology \
            src/cpu src/global src/gravity src/gravity/paris src/grid src/hydro \
            src/integrators src/io src/main.cpp src/main_tests.cpp src/mhd\
            src/model src/mpi src/old_cholla src/particles src/radiation src/radiation/alt \
            src/reconstruction src/riemann_solvers src/system_tests src/utils src/dust

SUFFIX ?= .$(TYPE).$(MACHINE)

CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

# Build a list of all potential object files so cleaning works properly
CLEAN_OBJS := $(subst .cpp,.o,$(CPPFILES)) \
              $(subst .cu,.o,$(GPUFILES))

# Check if it should include testing flags
ifeq ($(TEST), true)
  ADD_TEST_FLAGS = yes
  $(info Building Tests...)
  $(info )
  CPPFILES  := $(filter-out src/main.cpp,$(CPPFILES))
  # HACK
  # Set the build flags to debug. This is mostly to avoid the approximations
  # made by Ofast which break std::isnan and std::isinf which are required for
  # the testing
  BUILD = DEBUG
endif
ifeq ($(MAKECMDGOALS), tidy)
	ADD_TEST_FLAGS = yes
endif

# Set testing related lists and variables
ifeq ($(ADD_TEST_FLAGS), yes)
  # This is a test build so lets clear out Cholla's main file and set
  # appropriate compiler flags, suffix, etc
  SUFFIX    := $(strip $(SUFFIX)).tests
  LIBS      += -L$(GOOGLETEST_ROOT)/lib64 -pthread -lgtest -lhdf5_cpp
  TEST_FLAGS = -I$(GOOGLETEST_ROOT)/include
  CXXFLAGS += $(TEST_FLAGS)
  GPUFLAGS += $(TEST_FLAGS)
else
  # This isn't a test build so clear out testing related files
  CPPFILES := $(filter-out src/system_tests/% %_tests.cpp,$(CPPFILES))
  CPPFILES := $(filter-out src/utils/testing_utilities.cpp,$(CPPFILES))
  GPUFILES := $(filter-out src/system_tests/% %_tests.cu,$(GPUFILES))
endif

OBJS     := $(subst .cpp,.o,$(CPPFILES)) \
            $(subst .cu,.o,$(GPUFILES))

#-- Set default compilers and flags
CXX               ?= CC

CXXFLAGS_OPTIMIZE ?= -g -Ofast -std=c++17
GPUFLAGS_OPTIMIZE ?= -g -O3 -std=c++17

CXXFLAGS_DEBUG    ?= -g -O0 -std=c++17
ifdef HIPCONFIG
  GPUFLAGS_DEBUG    ?= -g -O0 -std=c++17
else
  GPUFLAGS_DEBUG    ?= -g -G -cudart shared -O0 -std=c++17 -ccbin=mpicxx
endif

BUILD             ?= OPTIMIZE

CXXFLAGS          += $(CXXFLAGS_$(BUILD))
GPUFLAGS          += $(GPUFLAGS_$(BUILD))

#-- Add flags and libraries as needed

CXXFLAGS += $(DFLAGS) -Isrc
GPUFLAGS += $(DFLAGS) -Isrc

ifeq ($(findstring -DPARIS,$(DFLAGS)),-DPARIS)
  ifdef HIPCONFIG
    CXXFLAGS += -I$(ROCM_PATH)/include/hipfft -I$(ROCM_PATH)/hipfft/include
    GPUFLAGS += -I$(ROCM_PATH)/include/hipfft -I$(ROCM_PATH)/hipfft/include
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

ifeq ($(findstring -DSUPERNOVA,$(DFLAGS)),-DSUPERNOVA)
    ifdef HIPCONFIG
	CXXFLAGS += -I$(ROCM_PATH)/include/hiprand -I$(ROCM_PATH)/hiprand/include
	GPUFLAGS += -I$(ROCM_PATH)/include/hiprand -I$(ROCM_PATH)/hiprand/include
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
  #GPUFLAGS  += -Wall
  LD        := $(CXX)
  LDFLAGS   := $(CXXFLAGS) -L$(ROCM_PATH)/lib
  LIBS      += -lamdhip64
else
  CUDA_INC  ?= -I$(CUDA_ROOT)/include
  CUDA_LIB  ?= -L$(CUDA_ROOT)/lib64 -lcudart
  CXXFLAGS  += $(CUDA_INC)
  GPUCXX    ?= nvcc
  GPUFLAGS  += --expt-extended-lambda -arch $(CUDA_ARCH) -fmad=false
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

.SUFFIXES: .cpp .cu .o

EXEC := bin/cholla$(SUFFIX)

# Get the git hash and setup macro to store a string of all the other macros so
# that they can be written to the save files
DFLAGS      += -DGIT_HASH='"$(shell git rev-parse --verify HEAD)"'
MACRO_FLAGS := -DMACRO_FLAGS='"$(DFLAGS)"'
DFLAGS      += $(MACRO_FLAGS)

# Setup variables for clang-tidy
LIBS_CLANG_TIDY     := $(subst -I/, -isystem /,$(LIBS))
LIBS_CLANG_TIDY     += -isystem $(MPI_ROOT)/include
CXXFLAGS_CLANG_TIDY := $(subst -I/, -isystem /,$(LDFLAGS))
GPUFLAGS_CLANG_TIDY := $(subst -I/, -isystem /,$(GPUFLAGS))
GPUFLAGS_CLANG_TIDY := $(filter-out -ccbin=mpicxx -fmad=false --expt-extended-lambda,$(GPUFLAGS))
GPUFLAGS_CLANG_TIDY += --cuda-host-only --cuda-path=$(CUDA_ROOT) -isystem /clang/includes
CPPFILES_TIDY := $(CPPFILES)
GPUFILES_TIDY := $(GPUFILES)

ifdef TIDY_FILES
  CPPFILES_TIDY := $(filter $(TIDY_FILES), $(CPPFILES_TIDY))
  GPUFILES_TIDY := $(filter $(TIDY_FILES), $(GPUFILES_TIDY))
endif

$(EXEC): prereq-build $(OBJS)
	mkdir -p bin/ && $(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
	eval $(EXTRA_COMMANDS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@

.PHONY: clean, clobber, tidy, format

format:
	tools/clang-format_runner.sh

tidy:
# Flags we might want
# - --warnings-as-errors=<string> Upgrade all warnings to error, good for CI
	clang-tidy --verify-config
	(time clang-tidy $(CLANG_TIDY_ARGS) $(CPPFILES_TIDY) -- $(DFLAGS) $(CXXFLAGS_CLANG_TIDY) $(LIBS_CLANG_TIDY)) > tidy_results_cpp.log 2>&1 & \
	(time clang-tidy $(CLANG_TIDY_ARGS) $(GPUFILES_TIDY) -- $(DFLAGS) $(GPUFLAGS_CLANG_TIDY) $(LIBS_CLANG_TIDY)) > tidy_results_gpu.log 2>&1 & \
	for i in 1 2; do wait -n; done

clean:
	rm -f $(CLEAN_OBJS)
	rm -rf googletest
	-find bin/ -type f -executable -name "cholla.*.$(MACHINE)*" -exec rm -f '{}' \;

clobber: clean
	-find bin/ -type f -executable -name "cholla*" -exec rm -f '{}' \;
	-find bin/ -type d -name "t*" -prune -exec rm -rf '{}' \;
	rm -rf bin/cholla.*tests*.xml

prereq-build:
	builds/prereq.sh build $(MACHINE)
prereq-run:
	builds/prereq.sh run $(MACHINE)
