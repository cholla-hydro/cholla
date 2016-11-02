EXEC   = cholla

OPTIMIZE =  -O2  

DIR = ./src
CFILES = $(wildcard $(DIR)/*.c)
CPPFILES = $(wildcard $(DIR)/*.cpp)
CUDAFILES = $(wildcard $(DIR)/*.cu)

OBJS = $(subst .c,.o,$(CFILES)) $(subst .cpp,.o,$(CPPFILES)) $(subst .cu,.o,$(CUDAFILES))

#To use MPI, MPI_FLAGS must be set to -DMPI_CHOLLA
#otherwise gcc/g++ will be used for serial compilation
#MPI_FLAGS =  -DMPI_CHOLLA

ifdef MPI_FLAGS
  CC	= mpicc
  CXX = mpicxx

  #MPI_FLAGS += -DSLAB
  #FFT_LIBS  += -lfftw3_mpi

  MPI_FLAGS += -DBLOCK
  MPI_FLAGS += -arch x86_64

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
#RECONSTRUCTION = -DPLMC
#RECONSTRUCTION = -DPPMP
RECONSTRUCTION = -DPPMC

#SOLVER = -DEXACT
SOLVER = -DROE
#SOLVER = -DHLLC

#INTEGRATOR = -DCTU 
INTEGRATOR = -DVL

#COOLING = -DCOOLING_CPU
#COOLING = -DCOOLING_GPU

INCL   = -I./ -I/usr/local/cuda/include -I/usr/local/include
NVLIBS = -L/usr/local/cuda/lib -lcuda -lcudart
LIBS   = -L/usr/local/lib -lm -lgsl -lhdf5


FLAGS = $(PRECISION) $(OUTPUT) $(RECONSTRUCTION) $(SOLVER) $(INTEGRATOR) $(COOLING) -DCUDA -DCUDA_ERROR_CHECK
CFLAGS 	  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) #-m64
CXXFLAGS  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) #-m64
NVCCFLAGS = $(FLAGS) -gencode arch=compute_30,code=sm_30 -fmad=false 
LDFLAGS	  = -Xlinker -rpath -Xlinker /usr/local/cuda/lib #-m64 -F/Library/Frameworks -framework CUDA


%.o:	%.c
		$(CC) $(CFLAGS)  $(INCL)  -c $< -o $@ 

%.o:	%.cpp
		$(CXX) $(CXXFLAGS)  $(INCL)  -c $< -o $@ 

%.o:	%.cu
		$(NVCC) $(NVCCFLAGS)  $(INCL)  -c $< -o $@ 

$(EXEC): $(OBJS) 
	 	 $(CXX) $(LDFLAGS) $(OBJS) $(LIBS) $(NVLIBS) -o $(EXEC) $(INCL) 

#$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

