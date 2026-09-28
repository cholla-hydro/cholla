SHELL = /usr/bin/env bash
#-- Set default include makefile
MACHINE ?= $(shell config/machine.sh)
TYPE    ?= hydro

include config/make.host.$(MACHINE)
include config/make.type.$(TYPE)

# CUDA_ARCH defaults to sm_70 if not set in make.host
CUDA_ARCH ?= sm_70

SUFFIX ?= .$(TYPE).$(MACHINE)

# a build directory is where we put temporary/intermediate build-artifacts
# -> more modern build systems (e.g. CMake/Meson) perform full out-of-source
#    builds in a directory like this
# -> an argument could be made that generation of the build directory should
#    be handled separately from this makefile (e.g. a configure script), but
#    that's probably an argument for another time
PROJDIR ?= $(shell pwd)
BUILDDIR ?= $(PROJDIR)/build

CPPFILES := $(shell find src/ -type f -name '*.cpp')
GPUFILES := $(shell find src/ -type f -name '*.cu')

# init a variable (to an empty string) for aggregating compiler flags passed to
# both the CXX compiler and the GPU compiler (hipcc or nvcc). This should NOT
# be set to a linker argument
COMMON_COMPILEFLAGS :=

# we must always use the C++ compiler-frontend to perform linking
# -> this is extemely important for automatically linking against
#    toolchain-specific runtime libraries. This manifests in 2 ways
#    1. when using MPI, we always use a compiler-frontend wrapper (e.g.
#       mpicxx). When this executable/script is used as a linker, it
#       automatically infers appropriate linker arguments for the
#       appropriate (toolchain-dependent) MPI runtime libraries.
#    2. Likewise, when a compiler-frontend is invoked to perform linking and
#       is passed the compilation option to enble handling of OpenMP
#       directives, the compiler-frontend infers linker arguments for the
#       appropriate OpenMP runtime library (e.g. libgomp, libiomp, libomp)
# -> while it's possible to infer these linker arguments (that's what cmake
#    does), that's not really feasible for us to do robustly.
LD := $(CXX)

# passed to linker (initially initialized to empty string)
LDFLAGS :=

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
  COMMON_COMPILEFLAGS += -isystem $(GOOGLETEST_ROOT)/include
else
  # This isn't a test build so clear out testing related files
  CPPFILES := $(filter-out src/system_tests/% %_tests.cpp,$(CPPFILES))
  CPPFILES := $(filter-out src/utils/testing_utilities.cpp,$(CPPFILES))
  GPUFILES := $(filter-out src/system_tests/% %_tests.cu,$(GPUFILES))
endif

ifeq ($(COVERAGE), true)
  CXXFLAGS += --coverage
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

# by passing `-include cholla_config.h` to the compiler, the C preprocessor
# acts as if the very first line of the source is `#include "cholla_config.h"`
#
# this is bad practice
COMMON_COMPILEFLAGS += -I./src -include cholla_config.h

ifeq ($(findstring -DPARIS,$(DFLAGS)),-DPARIS)
  ifdef HIPCONFIG
    COMMON_COMPILEFLAGS += -isystem $(ROCM_PATH)/include/hipfft -isystem $(ROCM_PATH)/hipfft/include
    LIBS += -L$(ROCM_PATH)/hipfft/lib -lhipfft
  else 
    ifdef NVIDIAMATH_ROOT
      # on a subset of CUDA platform, the NVIDIA MATH libraries are handled
      # separately from the rest of the core CUDA runtime libraries
      COMMON_COMPILEFLAGS += -isystem $(NVIDIAMATH_ROOT)/include
      LIBS += -L$(NVIDIAMATH_ROOT)/lib64 -lcufft
    else
      LIBS += -lcufft
    endif
  endif
  ifeq ($(findstring -DGRAVITY_5_POINTS_GRADIENT,$(DFLAGS)),-DGRAVITY_5_POINTS_GRADIENT)
    DFLAGS += -DPARIS_5PT
  else
    DFLAGS += -DPARIS_3PT
  endif
endif

ifeq ($(findstring -DFEEDBACK,$(DFLAGS)),-DFEEDBACK)
    ifdef HIPCONFIG
	COMMON_COMPILEFLAGS += -isystem $(ROCM_PATH)/include/hiprand -isystem $(ROCM_PATH)/hiprand/include
    endif
endif

ifeq ($(findstring -DHDF5,$(DFLAGS)),-DHDF5)
  COMMON_COMPILEFLAGS += -isystem $(HDF5_ROOT)/include
  LIBS += -L$(HDF5_ROOT)/lib -lhdf5
endif

ifeq ($(findstring -DMPI_CHOLLA,$(DFLAGS)),-DMPI_CHOLLA)
  # When using MPI, the c++ compiler frontend is always an mpi-specific wrapper
  # (e.g. mpicxx).
  # -> This wrapper automatically passes flags for the MPI-specific include
  #    directory to the wrapped compiler when compiling a C++ source file
  # -> we need to manually pass the location of this include-directory to the
  #    gpu compiler and to clang-tidy since neither tool uses the mpi-specific
  #    wrapper.
  GPUFLAGS += -isystem $(MPI_ROOT)/include
  CXXFLAGS_TIDY_EXTRA += -isystem $(MPI_ROOT)/include

  # TODO: is the following actually necessary? Because LD and CXX should both
  #       refer to the same compiler-frontend wrapper for a given MPI wrapper,
  #       these libraries should automatically be listed
  ifdef HIPCONFIG
     LIBS += -L$(MPI_ROOT)/lib -lmpi
  endif
endif

ifeq ($(findstring -DPARALLEL_OMP,$(DFLAGS)),-DPARALLEL_OMP)
  # there's an implicit assumption here that $(LD) (i.e. the command line tool
  # invoked for linker) is the same tool as $(CXX), the compiler front-end used
  # for compiling c++ files (we elaborate on these reasons when defining LD)
  #
  # It's important LDFLAGS is passed the EXACT same argument as CXXFLAGS so
  # that when the compiler-frontend is invoked as a linker, it links the
  # appropriate runtime library
  CXXFLAGS += -fopenmp
  LDFLAGS += -fopenmp

  # todo: eliminate -DPARALLEL_OMP from the configuration header and just
  #       have source/header files check whether _OPENMP is defined

  # ASIDE: strictly speaking, we could probably tease out the appropriate
  #        include-directory for "omp.h" (this isn't really practical for us
  #        to do). Because we don't currently do that clang-tidy typically
  #        hunts for the version of that header shipped with clang. This is
  #        probably ok since openmp is very standardized
endif

ifeq ($(findstring -DLYA_STATISTICS,$(DFLAGS)),-DLYA_STATISTICS)
  COMMON_COMPILEFLAGS += -isystem $(FFTW_ROOT)/include
  LIBS += -L$(FFTW_ROOT)/lib -lfftw3_mpi -lfftw3
endif

ifeq ($(findstring -DCOOLING_GRACKLE,$(DFLAGS)),-DCOOLING_GRACKLE)
  DFLAGS += -DCONFIG_BFLOAT_8
  DFLAGS += -DSCALAR
  COMMON_COMPILEFLAGS += -isystem $(GRACKLE_ROOT)/include
  LIBS += -L$(GRACKLE_ROOT)/lib -lgrackle
endif

ifeq ($(findstring -DCHEMISTRY_GPU,$(DFLAGS)),-DCHEMISTRY_GPU)
  DFLAGS += -DSCALAR
endif


# finalize CXXFLAGS and GPUFLAGS (among other things)
CXXFLAGS  += $(COMMON_COMPILEFLAGS)
GPUFLAGS  += $(COMMON_COMPILEFLAGS)
ifdef HIPCONFIG
  DFLAGS    += -DO_HIP
  CXXFLAGS  += $(HIPCONFIG)
  GPUCXX    ?= hipcc
  #GPUFLAGS  += -Wall
  LIBS      += -L$(ROCM_PATH)/lib -lamdhip64
else
  CUDA_INC  ?= -isystem $(CUDA_ROOT)/include
  CUDA_LIB  ?= -L$(CUDA_ROOT)/lib64 -lcudart
  CXXFLAGS  += $(CUDA_INC)
  GPUCXX    ?= nvcc
  GPUFLAGS  += --expt-extended-lambda -arch $(CUDA_ARCH) -fmad=false
  GPUFLAGS  += $(CUDA_INC)
  LIBS      += $(CUDA_LIB)
  DLINK	    := src/device_link.o
endif

.SUFFIXES: .cpp .cu .o

EXEC := bin/cholla$(SUFFIX)

# Get the git hash and setup macro to store a string of all the other macros so
# that they can be written to the save files
DFLAGS      += -DGIT_HASH=$(shell git rev-parse --verify HEAD)
MACRO_FLAGS := -DMACRO_FLAGS='$(DFLAGS)'
DFLAGS      += $(MACRO_FLAGS)

# Setup variables for clang-tidy
ifdef TIDY_FILES
  TARGET_TIDY_FILES := $(filter $(TIDY_FILES), $(CPPFILES)) \
		       $(filter $(TIDY_FILES), $(GPUFILES))
else
  TARGET_TIDY_FILES := $(CPPFILES) $(GPUFILES)
endif

$(EXEC): prereq-build $(OBJS)
	mkdir -p bin/
ifndef HIPCONFIG
	nvcc -dlink $(OBJS) -arch $(CUDA_ARCH) -o $(DLINK)
	$(LD) $(LDFLAGS) $(OBJS) $(DLINK) -o $(EXEC) $(LIBS)
else
	$(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
endif
	eval $(EXTRA_COMMANDS)

# here's a trick to ensure that src/cholla_config.h's recipe is rerun without declaring
# src/cholla_config.h to be PHONY (we shouldn't do that since the recipe makes the file)
# -> https://stackoverflow.com/a/60724811
.PHONY: FORCE
FORCE: ;

# this is the generated file that holds all of the DFLAGS
# -> even though this recipe is rerun every time make is invoked to build a target
#    with a direct or indirect dependency on src/cholla_config.h, we only mutate
#    src/cholla_config.h if the contents of the file changes
src/cholla_config.h: src/cholla_config.h.in FORCE
	tools/configure_file.py --clobber --input $< --output $@.tmp $(DFLAGS)
	cmp $@.tmp $@ || mv $@.tmp $@
	rm -rf $@.tmp

%.o: %.cpp src/cholla_config.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu src/cholla_config.h
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@

# construct a file named compile_commands.json
# -> the following command explicitly assumes that we're using CUDA
# -> While it would be nice to add support for HIP code, that would be tricky
#    for a few reasons and that is not something we've ever directly supported
$(BUILDDIR)/compile_commands.json: src/cholla_config.h tools/generate_compile_commands.py
	@mkdir -p $(BUILDDIR)
	tools/generate_compile_commands.py \
	    --compiler=/dummy/path/to/clang++ \
	    --output-file=$@-tmp-GPUONLY \
	    --sources $(GPUFILES) \
	    --directory=$(PROJDIR) \
	    --outputs-suffix=.o \
	    --strip-nvcc-flags \
	    -- $(GPUFLAGS) --cuda-host-only --cuda-path=$(CUDA_ROOT) -isystem /clang/includes  $(LIBS_CLANG_TIDY)

	tools/generate_compile_commands.py \
	    --compiler=$(CXX) \
	    --output-file=$@ \
	    --prepend-entries-from=$@-tmp-GPUONLY \
	    --sources $(CPPFILES) \
	    --directory=$(PROJDIR) \
	    --outputs-suffix=.o \
	    -- $(CXXFLAGS) $(CXXFLAGS_TIDY_EXTRA)
	rm $@-tmp-GPUONLY

.PHONY: clean, clobber, setup, tidy, format

format:
	tools/clang-format_runner.sh

setup: $(BUILDDIR)/compile_commands.json
	@# no-op

tidy: $(BUILDDIR)/compile_commands.json
# Flags we might want
# - --warnings-as-errors=<string> Upgrade all warnings to error, good for CI
	clang-tidy --verify-config
	@echo "Results from following clang-tidy command will be shown as they occur and will also be available in the 'tidy_results_$(TYPE).log' file"
	time clang-tidy -p ./build $(CLANG_TIDY_ARGS) $(TARGET_TIDY_FILES) 2>&1 | tee tidy_results_$(TYPE).log

clean:
	rm -f $(CLEAN_OBJS) $(DLINK) src/cholla_config.h
	rm -rf googletest
	rm -rf build
	-find bin/ -type f -executable -name "cholla.*.$(MACHINE)*" -exec rm -f '{}' \;
	-find src/ -type f -name "*.gcno" -delete
	-find src/ -type f -name "*.gcda" -delete

clobber: clean
	-find bin/ -type f -executable -name "cholla*" -exec rm -f '{}' \;
	-find bin/ -type d -name "t*" -prune -exec rm -rf '{}' \;
	rm -rf bin/cholla.*tests*.xml

prereq-build:
	config/prereq.sh build $(MACHINE)
prereq-run:
	config/prereq.sh run $(MACHINE)
