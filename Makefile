#-- Set default include makefile
MACHINE ?= $(shell builds/machine.sh)
TYPE    ?= hydro

include builds/make.host.$(MACHINE)
include builds/make.type.$(TYPE)

SUFFIX = .$(TYPE).$(MACHINE)

DIRS     := src src/gravity src/particles src/cosmology src/cooling
ifeq ($(findstring -DPARIS,$(POISSON_SOLVER)),-DPARIS)
  DIRS += src/gravity/paris
  DFLAGS += -DPARIS
  SUFFIX = .paris.$(MACHINE)
endif

CFILES   := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

OBJS     := $(subst .c,.o,$(CFILES)) \
            $(subst .cpp,.o,$(CPPFILES)) \
            $(subst .cu,.o,$(GPUFILES))

#-- Set default compilers and flags
CC                ?= cc
CXX               ?= CC

CFLAGS_OPTIMIZE    = -Ofast
CXXFLAGS_OPTIMIZE  = -Ofast -std=c++14
BUILD             ?= OPTIMIZE

CFLAGS             = $(CFLAGS_$(BUILD))
CXXFLAGS           = $(CXXFLAGS_$(BUILD))


#-- Add flags and libraries as needed

CFLAGS   += $(DFLAGS) -Isrc
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
  ifdef HIP_PLATFORM
     LIBS += -L$(MPI_ROOT)/lib -lmpi
  endif
endif

ifdef HIP_PLATFORM
  DFLAGS += -DO_HIP
  CXXFLAGS += -I$(ROCM_PATH)/include -Wno-unused-result
  CXXFLAGS += -D__HIP_PLATFORM_HCC__
  GPUCXX := hipcc
  GPUFLAGS += -g -Ofast -Wall --amdgpu-target=gfx906 -Wno-unused-variable \
              -Wno-unused-function -Wno-unused-result \
              -Wno-unused-command-line-argument -Wno-duplicate-decl-specifier \
              -std=c++14 -ferror-limit=1
  GPUFLAGS += -I$(ROCM_PATH)/include
  LD := $(GPUCXX)
  LDFLAGS += $(GPUFLAGS)
  LIBS += -L$(CRAYLIBS_X86_64) -L$(GCC_X86_64)/lib64 -lcraymath -lu
else
  GPUCXX := nvcc
  #GPUFLAGS += #--expt-extended-lambda -g -O3 -arch sm_70 -fmad=false
  GPUFLAGS += -g -O3 -arch sm_70 -fmad=false
  LD := $(CXX)
  LDFLAGS += $(CXXFLAGS)
  LIBS += -L$(CUDA_ROOT)/lib64 -lcudart
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
  CXXFLAGS += -I$(GRACKLE_ROOT)/include
  LIBS     += -L$(GRACKLE_ROOT)lib -lgrackle
endif

ifeq ($(findstring -DPARALLEL_OMP,$(DFLAGS)),-DPARALLEL_OMP)
  CXXFLAGS += -fopenmp
  ifdef HIP_PLATFORM
    LIBS += -lcraymp
  else
    LDFLAGS += -fopenmp
  endif
endif


.SUFFIXES: .c .cpp .cu .o

EXEC := cholla$(SUFFIX)

$(EXEC): prereq-build $(OBJS) 
	$(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
	eval $(EXTRA_COMMANDS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@

.PHONY: clean
	
clean:
	rm -f $(OBJS) 
	find . -type f -executable -name "cholla.*.$(MACHINE)" -exec rm -f '{}' \;

prereq-build:
	builds/prereq.sh build $(MACHINE)
prereq-run:
	builds/prereq.sh run $(MACHINE)

check : OUTPUT=-DOUTPUT
check : clean $(EXEC) prereq-run
	$(JOB_LAUNCH) ./cholla.$(TYPE).$(MACHINE) tests/regression/${TYPE}_input.txt
	builds/check.sh $(TYPE) tests/regression/${TYPE}_test.txt
